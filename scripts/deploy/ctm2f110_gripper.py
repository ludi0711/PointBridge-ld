"""CTM2F110 gripper control through xArm tool-port Modbus RTU passthrough."""

from __future__ import annotations

import time
from typing import Any


class CTM2F110Gripper:
    """Small wrapper around CTM2F110 gripper registers.

    The gripper is connected through the xArm tool port. The xArm SDK call used
    here is ``getset_tgpio_modbus_data(..., is_transparent_transmission=True)``.
    Position follows the vendor convention used in the local reference script:
    4 is closed, 100 is open.
    """

    REG_FINGER1_TORQUE = 11
    REG_FINGER1_SPEED = 12
    REG_FINGER2_TORQUE = 21
    REG_FINGER2_SPEED = 22
    REG_CLOSE = 40
    REG_OPEN = 41
    REG_SYNC_MODE = 42
    REG_STATUS = 43

    def __init__(
        self,
        robot: Any,
        *,
        slave_id: int = 1,
        host_id: int = 9,
        open_position: int = 100,
        close_position: int = 4,
        command_delay_s: float = 0.02,
    ) -> None:
        self.robot = robot
        self.slave_id = int(slave_id)
        self.host_id = int(host_id)
        self.open_position = int(open_position)
        self.close_position = int(close_position)
        self.command_delay_s = float(command_delay_s)

    @staticmethod
    def _modbus_crc(data: bytes) -> int:
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc

    @staticmethod
    def _clamp_int(value: int | float, lo: int, hi: int) -> int:
        return max(lo, min(hi, int(round(value))))

    def _request(self, func: int, addr: int, value: int) -> list[int]:
        payload = bytes(
            [
                self.slave_id,
                func,
                (addr >> 8) & 0xFF,
                addr & 0xFF,
                (value >> 8) & 0xFF,
                value & 0xFF,
            ]
        )
        crc = self._modbus_crc(payload)
        data = list(payload + bytes([crc & 0xFF, (crc >> 8) & 0xFF]))
        code, response = self.robot.getset_tgpio_modbus_data(
            data,
            host_id=self.host_id,
            is_transparent_transmission=True,
        )
        time.sleep(self.command_delay_s)
        if code != 0:
            raise RuntimeError(
                f"CTM2F110 Modbus request failed: code={code}, addr={addr}, value={value}"
            )
        return list(response or [])

    def _write_register(self, addr: int, value: int) -> None:
        self._request(0x06, addr, value)

    def _read_registers(self, addr: int, count: int) -> list[int]:
        response = self._request(0x03, addr, count)
        if len(response) < 5:
            return []

        byte_count = response[2] if len(response) > 2 else 0
        if byte_count <= 0 or len(response) < 3 + byte_count:
            return []

        values: list[int] = []
        for i in range(0, byte_count, 2):
            hi_idx = 3 + i
            lo_idx = hi_idx + 1
            if lo_idx >= len(response):
                continue
            values.append((response[hi_idx] << 8) | response[lo_idx])
        return values

    def configure(self, *, speed: int = 100, torque: int = 100) -> None:
        speed = self._clamp_int(speed, 1, 100)
        torque = self._clamp_int(torque, 0, 100)
        self._write_register(self.REG_SYNC_MODE, 1)
        self._write_register(self.REG_FINGER1_TORQUE, torque)
        self._write_register(self.REG_FINGER1_SPEED, speed)
        self._write_register(self.REG_FINGER2_TORQUE, torque)
        self._write_register(self.REG_FINGER2_SPEED, speed)

    def feedback(self, *, finger: int = 1) -> dict[str, int] | None:
        base_addr = 10 if int(finger) == 1 else 20
        values = self._read_registers(base_addr + 4, 6)
        if len(values) < 6:
            return None
        return {
            "position": values[0],
            "torque": values[1],
            "speed": values[2],
            "voltage": values[3],
            "current": values[4],
            "temperature": values[5],
        }

    def status(self) -> dict[str, bool] | None:
        values = self._read_registers(self.REG_STATUS, 2)
        if len(values) < 2:
            return None
        return {
            "is_running": bool(values[0]),
            "position_reached": bool(values[1]),
        }

    def stop(self) -> None:
        self._write_register(self.REG_CLOSE, 0)
        self._write_register(self.REG_OPEN, 0)

    def _start_direction(self, direction: int) -> None:
        if direction > 0:
            self._write_register(self.REG_CLOSE, 0)
            self._write_register(self.REG_OPEN, 1)
            return
        self._write_register(self.REG_OPEN, 0)
        self._write_register(self.REG_CLOSE, 1)

    def start_closing(self, *, speed: int = 30, torque: int = 60) -> None:
        self.configure(speed=speed, torque=torque)
        self._start_direction(-1)

    def start_opening(self, *, speed: int = 60, torque: int = 60) -> None:
        self.configure(speed=speed, torque=torque)
        self._start_direction(1)

    def set_position(
        self,
        position: int,
        *,
        speed: int = 100,
        torque: int = 100,
        wait: bool = True,
        timeout: float = 5.0,
        tolerance: int = 3,
    ) -> dict[str, int] | None:
        target = self._clamp_int(position, self.close_position, self.open_position)
        self.configure(speed=speed, torque=torque)

        current = self.feedback(finger=1)
        if current is None:
            if target >= self.open_position:
                direction = 1
            elif target <= self.close_position:
                direction = -1
            else:
                raise RuntimeError("Cannot move to an intermediate gripper position without feedback")
        else:
            current_pos = int(current["position"])
            if abs(current_pos - target) <= tolerance:
                self.stop()
                return current
            direction = 1 if current_pos < target else -1

        self._start_direction(direction)

        if not wait:
            return self.feedback(finger=1)

        last_feedback = None
        start = time.time()
        while time.time() - start < timeout:
            last_feedback = self.feedback(finger=1)
            if last_feedback is None:
                time.sleep(0.05)
                continue

            current_pos = int(last_feedback["position"])
            if abs(current_pos - target) <= tolerance:
                break
            if direction > 0 and current_pos >= target:
                break
            if direction < 0 and current_pos <= target:
                break
            time.sleep(0.05)

        self.stop()
        return last_feedback

    def open(
        self,
        *,
        speed: int = 100,
        torque: int = 100,
        wait: bool = True,
        timeout: float = 5.0,
    ) -> dict[str, int] | None:
        return self.set_position(
            self.open_position,
            speed=speed,
            torque=torque,
            wait=wait,
            timeout=timeout,
        )

    def close(
        self,
        *,
        speed: int = 100,
        torque: int = 100,
        wait: bool = True,
        timeout: float = 5.0,
    ) -> dict[str, int] | None:
        return self.set_position(
            self.close_position,
            speed=speed,
            torque=torque,
            wait=wait,
            timeout=timeout,
        )
