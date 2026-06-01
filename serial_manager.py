import serial
import threading

class SerialManager:

    def __init__(self, port="COM5", baudrate=115200):

        self.distance = 0
        self.motion = 0

        self.ser = serial.Serial(
            port,
            baudrate,
            timeout=0.1
        )

        self.running = True

        self.thread = threading.Thread(
            target=self.read_loop,
            daemon=True
        )

        self.thread.start()

    def read_loop(self):

        while self.running:

            try:
                line = self.ser.readline().decode().strip()

                if line.startswith("D:"):

                    parts = line.split(",")

                    self.distance = int(
                        parts[0].split(":")[1]
                    )

                    self.motion = int(
                        parts[1].split(":")[1]
                    )

            except:
                pass

    def send_servo(self, pan, tilt):

        cmd = f"P:{pan},T:{tilt}\n"

        try:
            self.ser.write(cmd.encode())
        except:
            pass

    def close(self):

        self.running = False

        if self.ser.is_open:
            self.ser.close()