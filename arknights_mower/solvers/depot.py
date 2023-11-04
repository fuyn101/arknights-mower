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
import csv
import concurrent.futures
import datetime
from ..utils.path import get_path


class DepotSolver(BaseSolver):
    """
    扫描仓库
    """

    def __init__(self, device: Device = None, recog: Recognizer = None) -> None:
        super().__init__(device, recog)
        sift = cv2.SIFT_create()
        #surf = cv2.xfeatures2d.SURF_create(600)
        self.detector = sift  # 检测器类型
        self.template_images_folder = str(get_path("@internal/dist/new"))  # 模板文件夹
        self.template_images = self.load_template_images(self.detector)  # 模板列表
        self.screenshot_dict = {}  # 所有截图的字典（尽量不重不漏）
        self.screenshot_count = 1  # 所有截图的列表的计数器
        self.image_set = []  # 每一张图像，数字，xy相对坐标
        self.matcher = cv2.FlannBasedMatcher(
            dict(algorithm=0, trees=2), dict(checks=30)
        )  # 初始化一个识别
        self.engine = RapidOCR(
            text_score=0.3,
            use_det=False,
            use_angle_cls=False,
            use_cls=False,
            use_rec=True,
        )
        self.circle_info = []
        self.match_results = []  # 匹配的结果
        self.translate_results = {}  # 好看的结果

    def load_template_images(self, detector) -> list:
        """
        加载给定文件夹中的模板图片。
        Returns:
            list: 包含模板图片的列表，每个元素是一个包含图片文件名和对应灰度图像的列表。

        """
        load_template_images = time.time()
        template_images = []

        for template_image_str in os.listdir(self.template_images_folder):
            template_file_path = os.path.join(
                self.template_images_folder, template_image_str
            )
            template_grey = cv2.imread(template_file_path, 0)
            _, des2 = detector.detectAndCompute(template_grey, None)
            template_images.append([template_image_str, des2])
        logger.info("Start: 仓库扫描预备，导入模板物品")
        logger.info(
            f"仓库扫描：模板图片加载完成，有{len(template_images)}张图片,用时{time.time() - load_template_images}"
        )

        return template_images

    def run(self) -> None:
        logger.info("Start: 仓库扫描")
        super().run()

    def transition(self) -> bool:
        logger.info("仓库扫描: 回到桌面")
        self.back_to_index()
        if self.scene() == Scene.INDEX:
            logger.info("仓库扫描: 从主界面点击仓库界面")
            self.tap_themed_element("index_warehouse")
            oldscreenshot = self.recog.gray
            self.recog.update()
            self.screenshot_dict[self.screenshot_count] = oldscreenshot  # 1 第一张图片
            logger.info(f"仓库扫描: 把第{self.screenshot_count}页保存进内存中等待识别")
            while True:
                self.swipe_only((1800, 450), (-300, 0), 200, 2)  # 滑动
                self.recog.update()
                newscreenshot = self.recog.gray
                similarity = self.compare_screenshot(
                    self.screenshot_dict[self.screenshot_count], newscreenshot
                )
                self.screenshot_count += 1  # 第二张图片
                if similarity < 70:
                    self.screenshot_dict[self.screenshot_count] = newscreenshot
                    logger.info(
                        f"仓库扫描: 把第{self.screenshot_count}页保存进内存中等待识别,相似度{similarity}"
                    )
                else:
                    logger.info("仓库扫描: 这大抵是最后一页了")
                    break
            logger.info(f"仓库扫描: 截图读取完了,有{len(self.screenshot_dict)}张截图")
            logger.info(f"仓库扫描: 开始计算裁切图像")
            for screenshot_times, screenshot_img in self.screenshot_dict.items():
                self.read_circle_and_cut_screenshot(screenshot_times, screenshot_img)
            logger.info(f"仓库扫描: 开始识别图像,需要识别{len(self.image_set)}个图像，很慢 别急")
            # for i in range(len(self.image_set)):
            #     self.match_once(self.image_set[i], self.template_images)
            self.match_results = parallel_match(self.image_set, self.template_images)
            self.translate(self.match_results)
            logger.info(f"仓库扫描: 识别结果{self.translate_results}")
            keys = ["日期"]
            keys.append(list(self.translate_results.keys()))
            values = [datetime.date.today()]
            values.append(list(self.translate_results.values()))
            csv_out = str(get_path("@app/tmp/depot.csv"))
            with open(csv_out, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(keys)  # 写入列名
                writer.writerow(values)  # 写入数据

        return True

    def read_circle_and_cut_screenshot(self, screenshot_times, screenshot_grey):
        """
        从灰度图像中检测圆。

        Args:
            screenshot_input (numpy.ndarray): 输入的灰度图像数组。
            output_path (str) : 截图输出路径

        Returns:
            List[Tuple[int, int]]: 包含检测到的圆信息的列表。每个圆由其圆心坐标和半径表示。
        """
        circle_info = []
        from_screenshot_read_circle_time = time.time()
        screenshot_out = np.copy(screenshot_grey)
        medianBlur = cv2.medianBlur(screenshot_grey, 5)
        def_Radius = 81
        dt = 0
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
        path = get_path("@app/screenshot/depot")

        if detected_circles is not None:
            detected_circles_int = np.uint16(np.around(detected_circles))

            for circle in detected_circles_int[0, :]:
                radius = circle[2]
                cv2.circle(
                    screenshot_out, (circle[0], circle[1]), radius, (0, 0, 255), 5
                )

                cv2.imwrite(f"{path}/screenshot-{screenshot_times}.png", screenshot_out)
                circle_info.append([circle[0], circle[1]])

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
            num_x = 48
            num_y = 137
            num_width = 110
            num_height = 50
            x = most_common_x // x_gap
            x = most_common_x - x * x_gap

            y = most_common_y // y_gap
            y = most_common_y - y * y_gap
            square_length = 187

            for x_t in range(8):
                for y_t in range(3):
                    x_value = x + x_t * x_gap
                    y_value = y + y_t * y_gap
                    if (
                        x_value + square_length // 2 < 1920
                        and x_value - square_length // 2 > 0
                    ):
                        if (
                            y_value + square_length // 2 < 1080
                            and y_value - square_length // 2 > 0
                        ):
                            square_top_left = (
                                x_value - square_length // 2,
                                y_value - square_length // 2,
                            )
                            square_bottom_right = (
                                x_value + square_length // 2,
                                y_value + square_length // 2,
                            )
                            cropped_square = screenshot_grey[
                                square_top_left[1] : square_bottom_right[1],
                                square_top_left[0] : square_bottom_right[0],
                            ]
                            cut_image = cropped_square[
                                num_y : num_y + num_height, num_x : num_x + num_width
                            ]
                            cut_image = cv2.resize(
                                cut_image,
                                (80 * 3, 40 * 3),
                                interpolation=cv2.INTER_LINEAR,
                            )
                            cut_image = cv2.threshold(
                                cut_image, 220, 255, cv2.THRESH_BINARY
                            )[1]

                            cv2.imwrite(
                                f"{path}/cropped_square-{screenshot_times}-{x_value},{y_value}.png",
                                cropped_square,
                            )

                            cv2.imwrite(
                                f"{path}/num_cut-{screenshot_times}-{x_value},{y_value}.png",
                                cut_image,
                            )
                            self.image_set.append(
                                [cropped_square, cut_image, x_value, y_value]
                            )
            logger.info(
                f"处理第{screenshot_times}张图像，用时{time.time()-from_screenshot_read_circle_time}"
            )
        else:
            logger.error(f"应该出错了")

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

    def compare_screenshot(self, image1, image2):
        keypoints1, descriptors1 = self.detector.detectAndCompute(image1, None)
        _, descriptors2 = self.detector.detectAndCompute(image2, None)

        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        similarity = len(good_matches) / len(keypoints1) * 100
        return similarity

    def translate(self, results):
        results_dict = template_dict
        for result in results:
            if result == None:
                pass
            else:
                key = result[0]
                self.translate_results[results_dict[key][0]] = [
                    result[2],
                    results_dict[key][1],
                    result[3],
                ]


def parallel_match(image_set, template_images):
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        results = list(
            executor.map(match_once, [(data, template_images) for data in image_set])
        )
    return results


engine = RapidOCR(
    text_score=0.3,
    use_det=False,
    use_angle_cls=False,
    use_cls=False,
    use_rec=True,
)


def match_once(data_template):
    """
    用于在给定的图片数据和图片集中寻找最佳匹配的函数。

    Args:
        data (tuple): 包含截取的方形区域和数字区域的图像数据。
            data[0]: 截取的方形区域的图像（numpy.ndarray）。
            data[1]: 数字区域的图像（numpy.ndarray）。
        template_images (list): 包含待匹配图像数据的列表。
            每个元素为一个tuple，包含图像名称和对应的图像数据。
            例如：[("image1.png", image_data1), ("image2.png", image_data2), ...]

    Returns:
        list: 包含匹配结果的列表。
                results[0]: 最佳匹配的图像名称（str）。
                results[1]: 识别出的数字（str）。
    """

    data, template_images = data_template
    sift = cv2.SIFT_create()
    #surf = cv2.xfeatures2d.SURF_create(600)
    detector = sift  # 检测器类型
    matcher = cv2.FlannBasedMatcher(
        dict(algorithm=0, trees=2), dict(checks=30)
    )  # 初始化一个识别

    (screenshot_cropped_square, num_cut, center_x, center_y) = data
    _, des1 = detector.detectAndCompute(screenshot_cropped_square, None)
    best_match_score, best_match_image = 0, None
    for template_images_name, des2 in template_images:
        matches = matcher.knnMatch(des1, des2, k=2)
        match_score = len([m for m, n in matches if m.distance < 0.7 * n.distance])
        if match_score > best_match_score:
            result_num = engine(num_cut)
            best_match_score, best_match_image = match_score, template_images_name
    logger.info(f"识别: {(center_x,center_y)}")
    try:
        match_result = [
            best_match_image[:-4],
            result_num[0][0][0],
            format_str(result_num[0][0][0]),
            (center_x, center_y),
        ]
        return match_result
    except Exception as e:
        match_result = ["空", "0", 0, (center_x, center_y)]


def format_str(s):
    try:
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

    except Exception as e:
        return "这张图片识别失败"
