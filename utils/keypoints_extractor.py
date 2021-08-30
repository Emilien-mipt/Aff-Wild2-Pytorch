import argparse
import ast
import os

import cv2
import face_alignment
import pandas


# def face_recognition
def landmarks_68_face_extractor(image, fa):
    try:
        dets = fa.get_landmarks_from_image(image)
        try:
            if len(dets) > 0:
                return dets[0].tolist(), True
        except TypeError:
            return None, False
    except ValueError:
        return None, False
    except RuntimeError:
        return None, False


def get_array(str_array):
    return ast.literal_eval(str_array)


def lm_csv_creator(curr_dir, fa, args):
    log_processed_pass = "./processed.txt"

    f = open(log_processed_pass, "r")

    processed_dirs = f.read().split("\n")
    print("Already processed dirs: ", processed_dirs)

    f = open(log_processed_pass, "a")

    dirslist = list({x for x in os.listdir(curr_dir) if x.find(".") < 0} - set(processed_dirs))

    print("Dir list: ", dirslist)
    print("*******************************************************************************************************")
    print(len(dirslist))

    for path in dirslist:
        print(f"Processing video {path} ...")

        imagelist = os.listdir(os.path.join(curr_dir, path))
        try:
            imagelist.remove("keypoints.csv")
        except ValueError:
            print("No keypoints file!")

        imagelist.sort(key=lambda x: int("".join(filter(str.isdigit, x))))

        df = pandas.DataFrame()

        for imagepath in imagelist:
            # print(imagepath)
            keypoints = {
                "frame": imagepath.split(".")[0],
                "detect": 1,
                "landmarks_68": [[0, 0] for i in range(68)],
            }
            # print(os.path.join(curr_dir, path, imagepath))
            im = cv2.imread(os.path.join(curr_dir, path, imagepath))
            landmarks_68, is_face = landmarks_68_face_extractor(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), fa)

            keypoints["detect"] = 1

            if is_face:
                keypoints.update({"landmarks_68": landmarks_68})

                if args.show:
                    for i in range(68):
                        # print(i, landmarks_68[i][0], landmarks_68[i][1])
                        im = cv2.circle(im, (int(landmarks_68[i][0]), int(landmarks_68[i][1])), 1, (255, 0, 0), 3)

                    cv2.imshow("q", im)
                    cv2.waitKey(3)

                df = df.append(keypoints, ignore_index=True)

            else:
                print(f"Face on image {os.path.join(curr_dir, path, imagepath)} has not been detected!")
                os.remove(os.path.join(curr_dir, path, imagepath))

        df.to_csv(os.path.join(curr_dir, path, "keypoints.csv"), index=False)
        print(f"Video {path} has been processed!")
        f.write("\n")
        f.write(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", default="", help="Path to image directory")
    parser.add_argument("-s", "--show", default=False, help="Show landmarks")
    args = parser.parse_args()
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device="cuda:0", face_detector="sfd")
    lm_csv_creator(args.dir, fa, args)
