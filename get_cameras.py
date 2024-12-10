import cv2


def list_ports() -> tuple:
    """
        Test the ports and returns a tuple with the available ports and the ones that are working.
    """

    is_working = True
    dev_port = 5
    working_ports = []
    available_ports = []

    while dev_port < 10:
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            is_working = False
            print(f"Port {dev_port} is not working.")
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print(
                    f"Port {dev_port} is working and reads images ({h} x {w})")
                working_ports.append(dev_port)
            else:
                print(
                    f"Port {dev_port} for camera ( {h} x {w}) is present but does not reads.")
                available_ports.append(dev_port)
        dev_port += 1
    return (available_ports, working_ports)


def main() -> None:
    list_ports()


if __name__ == "__main__":
    main()
