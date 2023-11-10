from ..utils import typealias as tp
from ..utils.device import Device
from ..utils.log import logger
from ..utils.recognize import Recognizer, Scene
from ..utils.solver import BaseSolver
from ..data import template_dict
import os
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR
import time
import re
from ..utils.path import get_path
import cv2
import os
import multiprocessing
import functools
import itertools
from .. import __rootdir__
from pathlib import Path

surf_arg = {"upright": True, "extended": False, "hessianThreshold": 100}


class DepotSolver(BaseSolver):
    """
    扫描仓库
    """

    def __init__(self, device: Device = None, recog: Recognizer = None) -> None:
        super().__init__(device, recog)
        sift = cv2.SIFT_create()
        # surf = cv2.xfeatures2d.SURF_create(**surf_arg)
        self.detector = sift  # 检测器类型

        self.numdict = []
        self.template_images_folder = str(get_path("@internal/dist/new"))  # 模板文件夹
        self.template_images = []  # 模板列表
        self.screenshot_dict = {}  # 所有截图的字典（尽量不重不漏）
        self.screenshot_count = 1  # 所有截图的列表的计数器

        self.image_set = []  # 每一张图像，数字，x y相对坐标
        self.matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=2), dict(checks=50)
        )  # 初始化一个识别
        self.crop_width = 70
        self.scale_factor = 0.85
        self.circle_info = []
        self.match_results = []  # 匹配的结果
        self.translate_results = {}  # 好看的结果
        self.num_template_images = []
        self.load_template_images()

    def load_template_images(self) -> None:
        """
        加载给定文件夹中的模板图片。
        Returns:
            list: 包含模板图片的列表，每个元素是一个包含图片文件名和对应灰度图像的列表。

        """
        load_template_images = time.time()

        logger.info("Start: 仓库扫描预备，导入模板物品")
        for template_image_str in os.listdir(self.template_images_folder):
            template_file_path = str(
                os.path.join(self.template_images_folder, template_image_str),
            )

            template_img = cv2.imread(template_file_path, cv2.IMREAD_COLOR)
            template_img = cv2.resize(
                template_img, None, fx=1 / self.scale_factor, fy=1 / self.scale_factor
            )
            self.template_images.append(
                [
                    template_image_str,
                    template_img[
                        template_img.shape[0] // 2
                        - self.crop_width : template_img.shape[0] // 2
                        + self.crop_width,
                        template_img.shape[1] // 2
                        - self.crop_width : template_img.shape[1] // 2
                        + self.crop_width,
                    ],
                ]
            )
        logger.info(
            f"仓库扫描：模板图片加载完成，有{len(self.template_images)}张图片,用时{time.time() - load_template_images}"
        )

    def run(self) -> None:
        logger.info("Start: 仓库扫描")
        super().run()

    def transition(self) -> bool:
        logger.info("仓库扫描: 回到桌面")
        self.back_to_index()
        if self.scene() == Scene.INDEX:
            logger.info("仓库扫描: 从主界面点击仓库界面")
            self.tap_themed_element("index_warehouse")

            oldscreenshot = self.recog.img
            oldscreenshot = cv2.cvtColor(oldscreenshot, cv2.COLOR_RGB2BGR)
            self.recog.update()
            self.screenshot_dict[self.screenshot_count] = oldscreenshot  # 1 第一张图片
            logger.info(f"仓库扫描: 把第{self.screenshot_count}页保存进内存中等待识别")
            while True:
                self.swipe_only((1800, 450), (-400, 0), 200, 2)  # 滑动
                self.recog.update()
                newscreenshot = self.recog.img
                newscreenshot = cv2.cvtColor(newscreenshot, cv2.COLOR_RGB2BGR)
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
            num_processes = 3  # 设置并行处理的进程数量
            logger.info(f"仓库扫描: 开始识别图像,需要识别{len(self.image_set)}个图像，很慢 别急")
            partial_match_template_for_item = functools.partial(
                _match_template_for_item, template_images=self.template_images
            )
            with multiprocessing.Pool(processes=num_processes) as pool:
                self.match_results = pool.map(
                    partial_match_template_for_item, self.image_set
                )
            self.translate(self.match_results)
            logger.info(
                f"仓库扫描: 识别完成,共{len(self.translate_results)}个结果,{self.translate_results}"
            )

        return True

    def read_circle_and_cut_screenshot(self, screenshot_times, screenshot_img):
        """
        从灰度图像中检测圆。

        Args:
            screenshot_input (numpy.ndarray): 输入的灰度图像数组。
            output_path (str) : 截图输出路径

        Returns:
            List[Tuple[int, int]]: 包含检测到的圆信息的列表。每个圆由其圆心坐标和半径表示。
        """
        from_screenshot_read_circle_time = time.time()

        circle_info = []

        screenshot_cut = np.copy(screenshot_img)
        screenshot_out = np.copy(screenshot_img)

        screenshot_grey = cv2.cvtColor(screenshot_img, cv2.COLOR_RGB2GRAY)
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
        get_path("@app/screenshot/depot").mkdir(exist_ok=True)
        path = get_path("@app/screenshot/depot")

        if detected_circles is not None:
            detected_circles_int = np.uint16(np.around(detected_circles))

            for circle in detected_circles_int[0, :]:
                radius = circle[2]
                cv2.circle(
                    screenshot_out, (circle[0], circle[1]), radius, (0, 0, 255), 5
                )

                circle_info.append([circle[0], circle[1]])
            cv2.imwrite(f"{path}/screenshot-{screenshot_times}.png", screenshot_out)

            x_counts = {}

            for coord in circle_info:
                x, _ = coord
                x_counts[x] = x_counts.get(x, 0) + 1
            most_common_x = max(x_counts, key=x_counts.get)

            x_gap = 233

            x = most_common_x // x_gap
            x = most_common_x - x * x_gap

            square_length = 187

            x_values = np.arange(x, x + 8 * x_gap, x_gap)
            y_values = np.arange(283, 283 + 3 * 286, 286)

            for x_value, y_value in itertools.product(x_values, y_values):
                if (
                    x_value + square_length // 2 < 1920
                    and x_value - square_length // 2 > 0
                ):
                    square_top_left = (
                        x_value - square_length // 2,
                        y_value - square_length // 2,
                    )
                    square_bottom_right = (
                        x_value + square_length // 2,
                        y_value + square_length // 2,
                    )
                    square = screenshot_cut[
                        square_top_left[1] : square_bottom_right[1],
                        square_top_left[0] : square_bottom_right[0],
                    ]
                    cropped_square = np.copy(
                        square[
                            square.shape[0] // 2
                            - self.crop_width : square.shape[0] // 2
                            + self.crop_width,
                            square.shape[1] // 2
                            - self.crop_width : square.shape[1] // 2
                            + self.crop_width,
                        ]
                    )
                    cut_image = np.copy(square[130:170, :])
                    self.image_set.append([cropped_square, cut_image, x_value, y_value])
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
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
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
                    results_dict[key][0],
                    results_dict[key][2],
                    result[2],
                    results_dict[key][1],
                    key,
                ]


engine = RapidOCR(
    text_score=0.3,
    use_det=False,
    use_angle_cls=False,
    use_cls=False,
    use_rec=True,
)


def load_num_template_images(num_template_dir):
    num_template_images = []
    for filename in os.listdir(num_template_dir):
        file_path = os.path.join(num_template_dir, filename)
        num_template_images.append([cv2.imread(file_path, 0), filename])
    return num_template_images


num_template_dir = str(Path(f"{__rootdir__}/resources/digits"))
num_template_images = load_num_template_images(num_template_dir)


def read_num(num_item_image, x, y, distance_threshold=5, threshold=0.8):
    # Crop the region of interest

    # Initialize variables
    num_right_bottom_coordinates = []
    num_list = []
    num_gray_item = np.copy(num_item_image)
    num_gray_item = cv2.cvtColor(num_gray_item, cv2.COLOR_BGR2GRAY)
    for num_template_image, num_template_label in num_template_images:
        # Template matching
        num_result = cv2.matchTemplate(
            num_gray_item, num_template_image, cv2.TM_CCOEFF_NORMED
        )

        # Get matching locations
        num_loc = np.where(num_result >= threshold)

        # Get template width and height
        num_template_width, num_template_height = num_template_image.shape[:2]

        # Loop through matching locations and draw rectangles on the item image
        for pt in zip(*num_loc[::-1]):
            num_top_left = pt
            num_bottom_right = (
                num_top_left[0] + num_template_height,
                num_top_left[1] + num_template_width,
            )

            # Check if coordinates are far enough from previous ones
            if all(
                np.linalg.norm(np.array(coord) - np.array(num_bottom_right))
                > distance_threshold
                for coord in num_right_bottom_coordinates
            ):
                # Store right bottom coordinates
                num_right_bottom_coordinates.append(num_bottom_right)
                num_number = num_template_label[6:-4]
                num_list.append([num_bottom_right[0], num_number, (x, y)])
                # Draw rectangle on the item image
                cv2.rectangle(num_item_image, num_top_left, num_bottom_right, (200), 2)

    # Sort and process num_list
    num_sorted = sorted(num_list, key=lambda x: x[0])

    num_str = ""
    for _, num_template_img, _ in num_sorted:
        num_temp = num_str
        num_all = num_temp + num_template_img
        num_str = num_all.replace("10000", "万").replace("dot", "..").replace("..", ".")
    # Save result image
    # cv2.imwrite(os.path.join(self.result_dir, f"{(x, y)}.png"), num_item_image)

    return num_str


def _match_template_for_item(args, template_images):
    best_match_score = 0
    best_template_filename = None
    best_img = None
    best_num_img=None
    item_img, num_img, x, y = args

    for template_filename, template_img in template_images:
        result = cv2.matchTemplate(template_img, item_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        if max_val > best_match_score:
            best_match_score = max_val
            best_num_img = num_img
            # result_num = engine(num_img)

            best_template_filename = template_filename
            best_img = template_img
    result_num = read_num(best_num_img, x, y)
    best_each = cv2.hconcat([item_img, best_img])
    path = get_path("@app/screenshot/depot")
    cv2.imwrite(
        f"{path}/two-img{best_template_filename}+{(x, y)}.png",
        best_each,
    )
    logger.debug(f"{best_template_filename}, {(x, y)}")
    match_results = [
        best_template_filename[:-4],
        result_num,
        # format_str(result_num[0][0][0]),
        format_str(result_num),
    ]
    return match_results


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
        logger.error(f"这张图片识别失败")
        return "这张图片识别失败"
