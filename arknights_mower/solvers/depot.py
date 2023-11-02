from ..utils import typealias as tp
from ..utils.device import Device
from ..utils.log import logger
from ..utils.recognize import RecognizeError, Recognizer, Scene
from ..utils.solver import BaseSolver
from ..data import template_dict
import os
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR
import time
import re

from ..utils.path import get_path


class DepotSolver(BaseSolver):
    """
    扫描仓库
    """

    def __init__(self, device: Device = None, recog: Recognizer = None) -> None:
        super().__init__(device, recog)
        sift = cv2.SIFT_create()
        # surf = cv2.xfeatures2d.SURF_create(600)
        self.detector = sift
        self.template_images_folder = str(get_path("@internal/dist/new"))  # 不知道怎么写
        self.template_images = self.load_template_images(
            self.template_images_folder, self.detector
        )

    def swipe_only(
        self,
        start: tp.Coordinate,
        movement: tp.Coordinate,
        duration: int = 100,
        interval: float = 1,
    ) -> None:
        """swipe only, no rebuild and recapture"""
        end = (start[0] + movement[0], start[1] + movement[1])
        self.device.swipe(start, end, duration=duration)
        if interval > 0:
            time.sleep(interval)

    def run(self) -> None:
        # if it touched
        self.touched = False

        logger.info("Start: 仓库扫描")
        super().run()

    def transition(self) -> bool:
        screenshot_list = []
        logger.info("仓库扫描: 回到桌面")
        self.back_to_index()
        if self.scene() == Scene.INDEX:
            self.tap_themed_element("index_warehouse")
            logger.info("仓库扫描: 从主界面点击仓库界面")
        screenshot_list.append(self.recog.gray)

        logger.info("仓库扫描: 把这页加入内存")

        oldscreenshot = self.recog.gray

        self.swipe_only([0, 450], [1900, 450], 100, 1)
        logger.info("仓库扫描: 拖动至下一页")
        self.recog.update()
        newscreenshot = self.recog.gray
        similarity = self.compare_images(newscreenshot, oldscreenshot)
        logger.info(f"上页和这页的相似度{similarity}")

        file_path = get_path("@app/screenshot/depot/new.png")

        file_path = str(file_path)

        cv2.imwrite(file_path, self.recog.gray)
        main_start_time = time.time()

        results = []

        results.extend(self.ocr_pages(self.recog.gray, file_path))

        matchs_dict = self.translate(results)
        logger.info(
            f"扫描结果 ：{len(matchs_dict)}{matchs_dict} 用时{time.time() - main_start_time}"
        )

    def load_template_images(self, template_images_folder, detecter):
        """
        加载给定文件夹中的模板图片。
        Args:
            template_images_folder(str) 包含模板图片的文件夹路径。
            detecter(Object) 探测器类型
        Returns:
            list: 包含模板图片的列表，每个元素是一个包含图片文件名和对应灰度图像的列表。

        """
        load_template_images = time.time()
        template_images = []

        for template_image_str in os.listdir(template_images_folder):
            template_file_path = os.path.join(
                template_images_folder, template_image_str
            )
            template_GRAY = cv2.imread(template_file_path, cv2.IMREAD_GRAYSCALE)
            _, des2 = detecter.detectAndCompute(template_GRAY, None)
            template_images.append([template_image_str, des2])

        logger.info(
            f"模板图片加载完成，有{len(template_images)}张图片,用时{time.time() - load_template_images}"
        )

        return template_images

    def from_screenshot_read_circle(self, screenshot_input, output_path):
        """
        从灰度图像中检测圆。

        Args:
            screenshot_input (numpy.ndarray): 输入的灰度图像数组。
            output_path (str) : 截图输出路径

        Returns:
            List[Tuple[int, int]]: 包含检测到的圆信息的列表。每个圆由其圆心坐标和半径表示。
        """
        from_screenshot_read_circle_time = time.time()
        screenshot_copy = np.copy(screenshot_input)
        medianBlur = cv2.medianBlur(screenshot_input, 5)
        def_Radius = 81
        dt = 0
        detected_circles = None
        detected_circles = cv2.HoughCircles(
            medianBlur,
            cv2.HOUGH_GRADIENT,
            dp=2,
            minDist=200,
            param1=100,
            param2=40,
            minRadius=def_Radius - dt,
            maxRadius=def_Radius + dt,
        )

        if detected_circles is not None:
            detected_circles_int = np.uint16(np.around(detected_circles))
            circle_info = []
            for circle in detected_circles_int[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]
                cv2.circle(screenshot_copy, center, radius, (0, 0, 255), 5)
                cv2.imwrite(output_path, screenshot_copy)
                circle_info.append([circle[0], circle[1]])
            logger.info(
                f"这张仓库截图查找到{len(circle_info)}个圆心,用时在{time.time()-from_screenshot_read_circle_time}秒"
            )
        x_counts = {}
        y_counts = {}
        for coord in circle_info:
            x, y = coord
            x_counts[x] = x_counts.get(x, 0) + 1
            y_counts[y] = y_counts.get(y, 0) + 1
        most_common_x = max(x_counts, key=x_counts.get)
        most_common_y = max(y_counts, key=y_counts.get)
        y_gap = 286
        x_gap = 233
        x = (most_common_x) // x_gap
        y = (most_common_y) // y_gap
        x = most_common_x - x * x_gap
        y = most_common_y - y * y_gap
        length = 187
        values = []
        for x_t in range(8):
            for y_t in range(3):
                x_value = x + x_t * x_gap
                y_value = y + y_t * y_gap
                if x_value + length // 2 < 1920 and x_value - length // 2 > 0:
                    if y_value + length // 2 < 1080 and y_value - length // 2 > 0:
                        values.append([x_value, y_value])
        logger.info(f"{values, len(values)}")

        return values  # 返回包含检测到的圆的图像路径列表

    def use_circle_cut_picture(
        self,
        screenshot_GREY_np,
        HoughCircle_24_int,
        num_x=48,
        num_y=137,
        num_width=110,
        num_height=50,
    ):
        """
        用于处理截图图像的函数。

        Args:
            screenshot_GREY_np (numpy.ndarray): 输入的灰度截图图像。
            HoughCircle_24_int (list): Hough圆变换检测到的圆的列表。
            num_x (int, optional): 数字区域左上角的x坐标。默认为70。
            num_y (int, optional): 数字区域左上角的y坐标。默认为135。
            num_width (int, optional): 数字区域的宽度。默认为90。
            num_height (int, optional): 数字区域的高度。默认为40。

        Returns:
            list: 包含截取的圆形区域和数字区域的图像列表。
        """
        path = get_path("@app/screenshot/depot")
        image_set = []
        for circle in HoughCircle_24_int:
            center_x, center_y = circle[0], circle[1]
            square_size = 186
            x_cor = max(0, center_x - square_size // 2)
            square_top_left = (x_cor, center_y - square_size // 2)
            square_bottom_right = (
                center_x + square_size // 2,
                center_y + square_size // 2,
            )
            cropped_square = screenshot_GREY_np[
                square_top_left[1] : square_bottom_right[1],
                square_top_left[0] : square_bottom_right[0],
            ]
            cut_image = cropped_square[
                num_y : num_y + num_height, num_x : num_x + num_width
            ]
            cut_image = cv2.resize(
                cut_image, (80 * 3, 40 * 3), interpolation=cv2.INTER_LINEAR
            )
            cut_image = cv2.threshold(cut_image, 220, 255, cv2.THRESH_BINARY)[1]

            cv2.imwrite(
                f"{path}/{center_x},{center_y}-cropped_square.png",
                cropped_square,
            )

            cv2.imwrite(
                f"{path}/{center_x},{center_y}-num_cut.png",
                cut_image,
            )
            image_set.append([cropped_square, cut_image, center_x, center_y])
        return image_set

    def format_str(self, s):
        logger.info(f"{s}要修改为")
        # 将连续两个以上的点替换为一个点
        s = re.sub(r"\.{2,}", ".", s)
        # 移除除了数字、点、和万之外的所有字符
        s = re.sub(r"[^\d万\.]", "", s)

        # 将全角数字转换为半角数字
        s = s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))

        if s == "":
            formatted_number = 0
            return formatted_number

        # 检查是否有万字符

        if "万" in s:
            # 如果是 a.b万 的形式，将格式化为 a.b*10000
            if "." in s:
                s = s.replace("万", "")
                integer_part, decimal_part = s.split(".")
                formatted_number = int(integer_part) * 10000 + int(decimal_part) * 1000
            # 如果格式是 a万  的形式格式化为 a*10000
            else:
                formatted_number = int(s.replace("万", "")) * 10000
        else:
            # 如果格式是 a.b 的形式且没有万这个字符，将格式化为 b
            if "." in s:
                formatted_number = int(s.split(".")[-1])
            # 如果没有小数点，返回原始输入
            else:
                formatted_number = int(s)

        return formatted_number

    def match_once(self, data, png_images):
        """
        用于在给定的图片数据和图片集中寻找最佳匹配的函数。

        Args:
            data (tuple): 包含截取的方形区域和数字区域的图像数据。
                data[0]: 截取的方形区域的图像（numpy.ndarray）。
                data[1]: 数字区域的图像（numpy.ndarray）。
            png_images (list): 包含待匹配图像数据的列表。
                每个元素为一个tuple，包含图像名称和对应的图像数据。
                例如：[("image1.png", image_data1), ("image2.png", image_data2), ...]

        Returns:
            list: 包含匹配结果的列表。
                    results[0]: 最佳匹配的图像名称（str）。
                    results[1]: 识别出的数字（str）。
        """

        matcher = cv2.FlannBasedMatcher(dict(algorithm=0, trees=2), dict(checks=30))
        engine = RapidOCR(
            text_score=0.3,
            use_det=False,
            use_angle_cls=False,
            use_cls=False,
            use_rec=True,
        )
        (screenshot_cropped_square, num_cut, center_x, center_y) = data
        _, des1 = self.detector.detectAndCompute(screenshot_cropped_square, None)
        best_match_score, best_match_image = 0, None
        for png_image_name, des2 in png_images:
            matches = matcher.knnMatch(des1, des2, k=2)
            match_score = len([m for m, n in matches if m.distance < 0.7 * n.distance])
            if match_score > best_match_score:
                result_num = engine(num_cut)
                best_match_score, best_match_image = match_score, png_image_name
        return [
            best_match_image[:-4],
            result_num[0][0][0],
            self.format_str(result_num[0][0][0]),
            center_x,
            center_y,
        ]

    def ocr_pages(self, screenshot_gray, file_path):
        circle_info = self.from_screenshot_read_circle(
            screenshot_gray, "after_" + file_path
        )
        results = self.ocr_one_page(screenshot_gray, circle_info, self.template_images)
        return results

    def ocr_one_page(self, screenshot_gray, circle_info, template_images):
        data = self.use_circle_cut_picture(screenshot_gray, circle_info)

        ocr_start_time = time.time()
        results = []
        for i in range(len(data)):
            result = self.ocr_worker(i, data, template_images)
            results.append(result)
        logger.info(f"匹配用时: {time.time() - ocr_start_time}")

        return results

    def ocr_worker(self, run_time, data, template_images):
        result = self.match_once(data[run_time], template_images)
        return result

    def translate(self, results):
        matchs_dict = {}
        results_dict = template_dict
        for result in results:
            key = result[0]
            matchs_dict[results_dict[key][0]] = [
                result[2],
                results_dict[key][1],
                (result[3], result[4]),
            ]
        return matchs_dict

    def compare_images(self, image1, image2):
        detector = self.detector
        keypoints1, descriptors1 = detector.detectAndCompute(image1, None)
        keypoints2, descriptors2 = detector.detectAndCompute(image2, None)
        flann_index_kdtree = 1
        index_params = dict(algorithm=flann_index_kdtree, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        similarity = len(good_matches) / len(keypoints1) * 100
        return similarity
