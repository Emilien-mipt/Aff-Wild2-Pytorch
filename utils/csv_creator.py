import argparse
import math
import os

import cv2
import face_alignment
import pandas

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device="cpu", face_detector="sfd")


# def face_recognition
def landmarks_68_face_extractor(image):
    try:
        dets = fa.get_landmarks_from_image(image)
        try:
            if len(dets) > 0:
                return dets[0], True

        except TypeError:
            return None, False
    except ValueError:
        return None, False
    except RuntimeError:
        return None, False


def lm_csv_creator(curr_dir):
    dirslist = [x for x in os.listdir(curr_dir) if x.find(".") < 0 and (x.find("left") >= 0 or x.find("right") >= 0)]
    print(dirslist)

    for path in dirslist:

        imagelist = [x.split(".")[0] for x in os.listdir(f"{curr_dir}/{path}/") if x.find(".png") >= 0]
        imagelist = [i + ".png" for i in sorted(imagelist, key=int)]
        df = pandas.DataFrame()

        for imagepath in imagelist:
            print(imagepath.split(".")[0])
            keypoints = {
                "frame": imagepath.split(".")[0],
                "detect": 1,
                "landmarks_68": [[0, 0] for i in range(68)],
                "nose": (0, 0),
                "mouth_right": (0, 0),
                "right_eye": (0, 0),
                "left_eye": (0, 0),
                "mouth_left": (0, 0),
            }
            print(f"processing: {curr_dir}/{path}/{imagepath}")
            landmarks_68, is_face = landmarks_68_face_extractor(
                cv2.cvtColor(cv2.imread(f"{curr_dir}/{path}/{imagepath}"), cv2.COLOR_BGR2RGB)
            )

            keypoints["detect"] = 1
            keypoints.update({"landmarks_68": landmarks_68})

            df = df.append(keypoints, ignore_index=True)

        df.to_csv(os.path.join(curr_dir, path, "keypoints.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", required=True, help="Path to image directory")
    args = parser.parse_args()
    lm_csv_creator(args.dir)
