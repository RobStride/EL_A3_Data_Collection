"""
Bus definitions.
电机总线定义

This file contains the definitions of the RobstrideBus class, which wraps around the socket-can interface.
"""


from dataclasses import dataclass
from functools import cached_property
import struct
import time
from typing import TypeAlias

import can
import numpy as np
from tqdm import tqdm

from .table import (
    MODEL_MIT_POSITION_TABLE,
    MODEL_MIT_VELOCITY_TABLE,
    MODEL_MIT_TORQUE_TABLE,
    MODEL_MIT_KP_TABLE,
    MODEL_MIT_KD_TABLE,
)
from .protocol import CommunicationType


Value: TypeAlias = int | float


@dataclass
class Motor:
    id: int
    model: str


class RobstrideBus:
    """
    A RobstrideBus allows to efficiently read and write to the attached motors.
    """
    def __init__(
        self,
        channel: str,
        motors: dict[str, Motor],
        calibration: dict[str, dict] | None = None,
        bitrate: int = 1000000,
    ):
        self.channel = channel
        self.motors = motors
        self.calibration = calibration
        self.bitrate = bitrate

        if self.calibration:
            print(f"Using calibration: {self.calibration}")
        else:
            print("WARNING: No calibration provided")

        self.channel_handler = None
        self._comm_success: int
        self._no_error: int

        # host ID needs to be greater than actuator ID to achieve optimal performance
        # here we simply set it to the maximum possible value
        self.host_id = 0xFF

        # 每个电机最近一次状态帧中的运行模式 (0=复位/未使能, 1=标定, 2=运行)
        # 用于检测"电机有应答但无力矩"的假使能状态
        self.last_status_mode: dict[str, int] = {}

    def __len__(self):
        return len(self.motors)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"    Channel: '{self.channel}',\n"
            f"    Motors: \n{self.motors},\n"
            ")',\n"
        )

    def __del__(self):
        if self.is_connected:
            self.disconnect()

    @cached_property
    def models(self) -> list[str]:
        return [m.model for m in self.motors.values()]

    @cached_property
    def ids(self) -> list[int]:
        return [m.id for m in self.motors.values()]

    @property
    def is_connected(self) -> bool:
        """bool: `True` if the underlying CAN bus is open."""
        return self.channel_handler is not None

    def connect(self, handshake: bool = True) -> None:
        """Open the serial port and initialise communication.
        """
        if self.is_connected:
            raise Exception(
                f"{self.__class__.__name__}('{self.channel}') is already connected. "
                f"Do not call `{self.__class__.__name__}.connect()` twice."
            )

        self.channel_handler = can.interface.Bus(
            interface="socketcan",
            channel=self.channel,
            bitrate=self.bitrate,
        )
        self._flush_rx_buffer()
        print(f"{self.__class__.__name__} connected.")

    @staticmethod
    def _log_fault_frame(motor: str, fault_value: int, warning_value: int) -> None:
        """Decode and print fault/warning bits from a FAULT_REPORT frame."""
        if fault_value & (1 << 0):
            print(f"FAULT: {motor} overtemperature")
        if fault_value & (1 << 1):
            print(f"FAULT: {motor} drive gate fault")
        if fault_value & (1 << 2):
            print(f"FAULT: {motor} undervoltage")
        if fault_value & (1 << 3):
            print(f"FAULT: {motor} overvoltage")
        if fault_value & (1 << 7):
            print(f"FAULT: {motor} encoder uncalibrated")
        if warning_value & (1 << 0):
            print(f"WARNING: {motor} overtemperature")
        if warning_value & (1 << 14):
            print(f"WARNING: {motor} stall current")

    def _flush_rx_buffer(self) -> None:
        """Drain any stale frames sitting in the CAN receive buffer."""
        flushed = 0
        while True:
            frame = self.channel_handler.recv(timeout=0.005)
            if frame is None:
                break
            flushed += 1
        if flushed:
            print(f"Flushed {flushed} stale CAN frame(s).")

    def disconnect(self, disable_torque: bool = True) -> None:
        """Close the serial port (optionally disabling torque first).
        """
        if not self.is_connected:
            raise Exception(
                f"{self.__class__.__name__}('{self.channel}') is not connected. "
                f"Try running `{self.__class__.__name__}.connect()` first."
            )

        if disable_torque:
            for motor in self.motors:
                try:
                    self.disable(motor)
                except Exception as e:
                    print(f"WARNING: Failed to disable {motor}: {e}")
            print("Torque disabled for all motors.")

        self.channel_handler.shutdown()
        self.channel_handler = None
        print(f"{self.__class__.__name__} disconnected.")

    @classmethod
    def scan_channel(cls, channel: str, start_id: int = 1, end_id: int = 255) -> dict[int, list[int]]:
        """Probe channel and list responding IDs.
        """
        bus = cls(channel, {})
        bus.connect(handshake=False)

        device_ids = {}

        for device_id in tqdm(range(start_id, end_id), desc="Scanning channel"):
            # print("\rTesting device ID:", device_id, end="")
            response = bus.ping_by_id(device_id, timeout=0.1)
            # print(f"Response: {response}")
            if response is not None:
                tqdm.write(f"Motors found for {device_id=}: {response}")
                device_ids[device_id] = list(response)

        bus.disconnect()
        return device_ids

    def read(self, motor: str, parameter_type: tuple[int, np.dtype, str]) -> Value:
        """Read a parameter from the motor.
        """
        device_id = self.motors[motor].id
        param_id, param_dtype, param_name = parameter_type

        data = struct.pack("<HHL", param_id, 0x00, 0x00)
        self.transmit(CommunicationType.READ_PARAMETER, self.host_id, device_id, data)
        response = self.receive_read_frame()

        match param_dtype:
            case np.uint8:
                value, _, _ = struct.unpack("<BBH", response)
            case np.int8:
                value, _, _ = struct.unpack("<bBH", response)
            case np.uint16:
                value, _ = struct.unpack("<hH", response)
            case np.int16:
                value, _ = struct.unpack("<hH", response)
            case np.uint32:
                value, = struct.unpack("<L", response)
            case np.int32:
                value, = struct.unpack("<l", response)
            case np.float32:
                value, = struct.unpack("<f", response)
            case _:
                raise ValueError(f"Unsupported parameter type of {param_name}: {param_dtype}")

        return value

    def write(self, motor: str, parameter_type: tuple[int, np.dtype, str], value: Value) -> None:
        """Write a value to a single motor's parameter.
        """
        device_id = self.motors[motor].id
        param_id, param_dtype, param_name = parameter_type

        match param_dtype:
            case np.uint8:
                value_buffer = struct.pack("<BBH", value, 0, 0)
            case np.int8:
                value_buffer = struct.pack("<bBH", value, 0, 0)
            case np.uint16:
                value_buffer = struct.pack("<HH", value, 0)
            case np.int16:
                value_buffer = struct.pack("<hH", value, 0)
            case np.uint32:
                value_buffer = struct.pack("<L", value)
            case np.int32:
                value_buffer = struct.pack("<l", value)
            case np.float32:
                value_buffer = struct.pack("<f", value)
            case _:
                raise ValueError(f"Unsupported parameter type of {param_name}: {param_dtype}")

        data = struct.pack("<HH", param_id, 0x00) + value_buffer

        self.transmit(CommunicationType.WRITE_PARAMETER, self.host_id, device_id, data)
        self.receive_status_frame(motor)

    # RobstrideMethods

    def transmit(
        self,
        communication_type: int,
        extra_data: int,
        device_id: int,
        data: bytes = b"\x00\x00\x00\x00\x00\x00\x00\x00",
    ):
        """
        Transmit data to the motor.
        Args:
            communication_type (int): Type of communication (通信类型).
            extra_data (int): Secondary data field (数据区 2).
            device_id (int): ID of the target actuator (目标地址).
            data (bytes): Primary data field (数据区 1).
        """

        assert (communication_type >= 0) and (communication_type <= 0x1F), "Communication type out of range"
        assert (extra_data >= 0) and (extra_data <= 0xFFFF), "Extra data out of range"
        assert (device_id > 0) and (device_id <= 0xFF), "Device ID out of range"
        assert len(data) <= 8, "Data length exceeds maximum size of 8 bytes"

        # build can fields
        ext_id = (communication_type << 24) | (extra_data << 8) | (device_id)
        dlc = len(data)

        frame = can.Message(
            arbitration_id=ext_id,
            is_extended_id=True,
            dlc=dlc,
            data=data,
        )

        self.channel_handler.send(frame)

    def receive(self, timeout: float | None = None) -> tuple[int, int, int, bytes] | None:
        """
        Receive a response from the motor.

        The CAN Extended ID field is separated into three parts:
        - Communication type (bit 28~24)
        - Extra data (bit 23~8)
        - Device ID (bit 7~0)

        CAN ID fields are Little Endian, and the data field is Big Endian.

        Returns:
            tuple: (communication_type, extra_data, device_id, data)
        """

        start_time = time.time()

        while timeout is None or time.time() - start_time < timeout:
            frame = self.channel_handler.recv(timeout=timeout)

            if not frame:
                print("WARNING: Received no response from the motor")
                return None

            if not frame.is_extended_id:
                # communication type 0 (device ID) will be recognized as a non-extended ID frame
                # this is common while the motor drops off and reconnects, it will send this frame
                # on the bus
                continue

            break

        communication_type = (frame.arbitration_id >> 24) & 0x1F
        extra_data = (frame.arbitration_id >> 8) & 0xFFFF
        host_id = frame.arbitration_id & 0xFF

        return communication_type, extra_data, host_id, frame.data

    def receive_status_frame(self, motor: str, max_retries: int = 5) -> tuple[float, float, float, float]:
        """
        Receives the response frame and update the status of the motor.
        Retries on unexpected frame types (e.g. stale GET_DEVICE_ID broadcasts)
        and warning-only fault frames.

        Returns:
            tuple: (position, velocity, torque, temperature)
        """
        for attempt in range(max_retries):
            received_frame = self.receive(timeout=0.1)
            if not received_frame:
                raise RuntimeError(f"No response from the motor {motor}")
            communication_type, extra_data, host_id, data = received_frame

            if communication_type not in (CommunicationType.OPERATION_STATUS, CommunicationType.FAULT_REPORT):
                print(f"WARNING: Unexpected communication type {communication_type} for {motor}, retrying ({attempt+1}/{max_retries})")
                continue

            if communication_type == CommunicationType.FAULT_REPORT:
                fault_value, warning_value = struct.unpack("<LL", data)
                self._log_fault_frame(motor, fault_value, warning_value)
                if fault_value != 0:
                    raise RuntimeError(
                        f"Motor {motor} has critical fault (fault=0x{fault_value:08X}, warning=0x{warning_value:08X})"
                    )
                print(f"WARNING: {motor} sent warning-only fault frame, retrying ({attempt+1}/{max_retries})")
                continue

            # 校验反馈帧确实来自我们查询的电机。总线故障风暴时接收队列会错位,
            # 若把别的电机的位置当成本电机的读数, 上层会把错乱角度发给从臂导致猛甩。
            device_id = (extra_data >> 0) & 0xFF
            if device_id != self.motors[motor].id:
                print(
                    f"WARNING: {motor} got frame from device {device_id} "
                    f"(expected {self.motors[motor].id}), discarding and retrying ({attempt+1}/{max_retries})"
                )
                continue

            break
        else:
            raise RuntimeError(
                f"Failed to receive a valid status frame from {motor} after {max_retries} retries "
                f"(last communication_type={communication_type})"
            )

        model = self.motors[motor].model
        # unpack the extra data field
        # since we already pre-unpacked the extra data field out, the shifting
        # will be 8 less than the number specified in the datasheet
        status_mode = (extra_data >> 14) & 0x03
        self.last_status_mode[motor] = status_mode
        status_uncalibrated = (extra_data >> 13) & 0x01
        status_stall = (extra_data >> 12) & 0x01
        status_magnetic_encoder_fault = (extra_data >> 11) & 0x01
        status_overtemperature = (extra_data >> 10) & 0x01
        status_overcurrent = (extra_data >> 9) & 0x01
        status_undervoltage = (extra_data >> 8) & 0x01

        if status_uncalibrated:
            print(f"WARNING: {motor} is uncalibrated")
        if status_stall:
            print(f"WARNING: {motor} is stalled")
        if status_magnetic_encoder_fault:
            print(f"WARNING: {motor} has a magnetic encoder fault")
        if status_overtemperature:
            print(f"WARNING: {motor} is overtemperature")
        if status_overcurrent:
            print(f"WARNING: {motor} is overcurrent")
        if status_undervoltage:
            print(f"WARNING: {motor} is undervoltage")

        # unpack the data
        position_u16, velocity_u16, torque_i16, temperature_u16 = struct.unpack(">HHHH", data)

        # normalize the data
        position = (float(position_u16) / 0x7FFF - 1.) * MODEL_MIT_POSITION_TABLE[model]
        velocity = (float(velocity_u16) / 0x7FFF - 1.) * MODEL_MIT_VELOCITY_TABLE[model]
        torque = (float(torque_i16) / 0x7FFF - 1.) * MODEL_MIT_TORQUE_TABLE[model]
        temperature = float(temperature_u16) * 0.1
        return position, velocity, torque, temperature

    def receive_read_frame(self) -> bytes:
        """
        Receive a parameter read response from the motor.
        """
        communication_type, extra_data, host_id, data = self.receive()

        # unpack the extended ID
        assert communication_type == CommunicationType.READ_PARAMETER, (
            f"Invalid communication type, got {communication_type}"
        )

        return data[4:]

    def ping_by_id(self, device_id: int, timeout: float | None = None):
        """
        Ping the motor by ID.
        """
        self.transmit(CommunicationType.GET_DEVICE_ID, self.host_id, device_id)
        response = self.receive(timeout)
        if not response:
            return None
        communication_type, device_id, check, uuid = response
        print(f"ID: {device_id}, UUID: {uuid}")
        return device_id, uuid

    def read_id(self, motor: str, timeout: float | None = None):
        """
        Read the ID of the motor.
        """
        device_id = self.motors[motor].id
        response = self.ping_by_id(device_id, timeout)
        return response

    def enable(self, motor: str):
        """
        Enable the motor and verify it actually entered run mode.

        电机在锁存故障/异常状态下可能对 ENABLE 有应答但并未真正使能
        (有应答无力矩), 因此必须校验状态帧中的运行模式位。
        """
        self._flush_rx_buffer()
        device_id = self.motors[motor].id
        self.transmit(CommunicationType.ENABLE, self.host_id, device_id)
        self.receive_status_frame(motor)

        if self.last_status_mode.get(motor) != 2:
            # ACK 帧未确认进入运行模式, 再发一帧零力矩 MIT 帧确认
            self.write_operation_frame(motor, position=0, kp=0, kd=0)
            self.receive_status_frame(motor)
        if self.last_status_mode.get(motor) != 2:
            raise RuntimeError(
                f"Motor {motor} did not enter run mode after ENABLE "
                f"(mode={self.last_status_mode.get(motor)}, 0=reset 1=cali 2=run), torque is OFF"
            )

    def disable(self, motor: str):
        """
        Disable the motor.
        """
        self._flush_rx_buffer()
        device_id = self.motors[motor].id
        self.transmit(CommunicationType.DISABLE, self.host_id, device_id)
        self.receive_status_frame(motor)

    def write_id(self, motor: str, new_id: int):
        """
        Write the ID of the motor.
        """
        device_id = self.motors[motor].id
        self.transmit(CommunicationType.SET_CAN_ID, new_id, device_id)
        response = self.receive()
        if not response:
            return None
        communication_type, device_id, check, uuid = response
        print(f"new ID: {device_id}, UUID: {uuid}")

        self.motors[motor].id = new_id
        return device_id, uuid

    def write_operation_frame(
        self,
        motor: str,
        position: float,
        kp: float,
        kd: float,
        velocity: float = 0,
        torque: float = 0,
    ):
        """
        Send an MIT frame to the motor.

        Args:
            motor (str): Motor name.
            position (float): Target position of the motor.
            kp (float): Proportional gain.
            kd (float): Derivative gain.
            velocity (float): Feedforward velocity of the motor.
            torque (float): Feedforward torque of the motor.
        """
        device_id = self.motors[motor].id
        model = self.motors[motor].model

        if self.calibration:
            # convert to raw motor frame
            calibration = self.calibration[motor]
            position = position * calibration["direction"] + calibration["homing_offset"]
            velocity = velocity * calibration["direction"]
            torque = torque * calibration["direction"]

        position = np.clip(position, -MODEL_MIT_POSITION_TABLE[model], MODEL_MIT_POSITION_TABLE[model])
        position_u16 = int(((position / MODEL_MIT_POSITION_TABLE[model]) + 1.) * 0x7FFF)
        position_u16 = np.clip(position_u16, 0x0, 0xFFFF)
        velocity = np.clip(velocity, -MODEL_MIT_VELOCITY_TABLE[model], MODEL_MIT_VELOCITY_TABLE[model])
        velocity_u16 = int(((velocity / MODEL_MIT_VELOCITY_TABLE[model]) + 1.) * 0x7FFF)
        velocity_u16 = np.clip(velocity_u16, 0x0, 0xFFFF)
        kp = np.clip(kp, 0., MODEL_MIT_KP_TABLE[model])
        kp_u16 = int((kp / MODEL_MIT_KP_TABLE[model]) * 0xFFFF)
        kd = np.clip(kd, 0., MODEL_MIT_KD_TABLE[model])
        kd_u16 = int((kd / MODEL_MIT_KD_TABLE[model]) * 0xFFFF)
        torque_u16 = int((torque / MODEL_MIT_TORQUE_TABLE[model] + 1.) * 0x7FFF)
        torque_u16 = np.clip(torque_u16, 0x0, 0xFFFF)

        data = struct.pack(">HHHH", position_u16, velocity_u16, kp_u16, kd_u16)

        self.transmit(CommunicationType.OPERATION_CONTROL, torque_u16, device_id, data)

    def read_operation_frame(self, motor: str) -> tuple[float, float, float, float]:
        """
        Receive the MIT status frame from the motor.

        Returns:
            tuple: (position, velocity, torque, temperature)
        """
        # receive the status frame
        status = self.receive_status_frame(motor)
        position, velocity, torque, temperature = status

        if self.calibration:
            calibration = self.calibration[motor]
            position = (position - calibration["homing_offset"]) * calibration["direction"]
            velocity = velocity * calibration["direction"]
            torque = torque * calibration["direction"]
            temperature = temperature

        return position, velocity, torque, temperature
