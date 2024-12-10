import cv2
import os


def main(fps):
    shape = cv2.imread("C:\\images\\1.png").shape
    length = len(os.listdir("C:\\images"))

    video_writer = cv2.VideoWriter("pysics-3.2-test.mp4", -1, fps, (shape[1], shape[0]))

    for n in range(length):
        img = cv2.imread(f"C:\\images\\{n}.png")
        video_writer.write(img)

    video_writer.release()
    print("Video process done.")


if __name__ == "__main__":
    main(60)
