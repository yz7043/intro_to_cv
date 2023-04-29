import cv2
import os

def convert_to_edge_img(root_dir, save_dir):
    imgs = os.listdir(root_dir)
    for img in imgs:
        read_dir = os.path.join(root_dir, img)
        output_dir = os.path.join(save_dir, img)
        img = cv2.imread(read_dir)
        edge_img = cv2.Canny(img, 150, 200)
        edge_rbg_img = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(output_dir, edge_rbg_img)

if __name__ == "__main__":
    convert_to_edge_img("edge2rgb/trainB/", "edge2rgb/trainA/")