import numpy as np
from PIL import Image
import pylab as pl
import scipy.cluster.vq as vq


def rgb2ycbcr(img):
    """
    将RGB图像转换为YCbCr格式，与Matlab版本效果一致

    参数:
        img: np.ndarray， dtype为uint8的RGB图像，像素值范围[0,255]

    返回:
        np.ndarray: 转换后的YCbCr图像，dtype为uint8
    """
    # RGB到YCbCr的转换矩阵（归一化处理）
    conversion_matrix = np.array(
        [
            [65.481, 128.553, 24.966],
            [-37.797, -74.203, 112],
            [112, -93.786, -18.214]
        ]
    ) / 255.0

    # 各通道偏移量
    offsets = [16, 128, 128]

    # 确保像素值不超过255
    img = np.minimum(img, 255)

    # 初始化结果数组
    result_img = np.zeros(img.shape, dtype=np.float64)

    # 逐通道计算转换结果
    for i in range(3):
        result_img[:, :, i] = (
                img[:, :, 0] * conversion_matrix[i, 0] +
                img[:, :, 1] * conversion_matrix[i, 1] +
                img[:, :, 2] * conversion_matrix[i, 2] +
                offsets[i]
        )

    # 四舍五入并转换为uint8类型
    result_img = np.round(result_img)
    return np.require(result_img, dtype=np.uint8)


def colorsegms(segm, img):
    """
    根据原图对分割结果进行着色，每个分割区域使用该区域内的平均颜色

    参数:
        segm: np.ndarray，分割图像（标签矩阵）
        img: np.ndarray或Image对象，原始RGB图像

    返回:
        np.ndarray: 着色后的分割图像
    """
    # 将图像转换为numpy数组并复制，避免修改原图
    img = np.asarray(img).copy()

    # 检查分割图与原图尺寸是否一致
    if segm.shape[0:2] != img.shape[0:2]:
        raise ValueError("分割图像与原始图像的尺寸不一致")

    # 分离RGB通道
    r_channel = img[:, :, 0]
    g_channel = img[:, :, 1]
    b_channel = img[:, :, 2]

    # 获取所有分割标签（假设标签从1开始）
    max_label = int(np.max(segm.flatten()))

    # 对每个分割区域计算平均颜色并填充
    for label in range(1, max_label + 1):
        # 获取当前标签对应的像素索引
        mask = (segm == label)

        # 计算该区域的平均RGB值并填充
        r_channel[mask] = np.mean(r_channel[mask].flatten())
        g_channel[mask] = np.mean(g_channel[mask].flatten())
        b_channel[mask] = np.mean(b_channel[mask].flatten())

    return img


def getfeatures(img, stepsize, follow_matlab=True):
    """
    从图像中提取特征，每个特征包含颜色信息和位置信息

    参数:
        img: np.ndarray或Image对象，输入图像
        stepsize: 滑动窗口的步长
        follow_matlab: 是否遵循Matlab的索引习惯（1-based）

    返回:
        X: np.ndarray，特征矩阵，每列是一个特征向量（4维）
        L: dict，包含特征提取的位置信息（窗口大小、步长等）
    """
    winsize = 7  # 滑动窗口大小（确保为奇数）

    # 检查步长是否合理
    if stepsize > winsize:
        raise ValueError("步长不能大于窗口大小")

    # 将图像转换为numpy数组
    img = np.asarray(img)

    # 检查图像格式是否符合要求（uint8类型的彩色图像）
    if img.dtype != np.uint8 or len(img.shape) != 3:
        raise ValueError("仅支持uint8类型的彩色图像")

    # 将RGB图像转换为YCbCr格式（用于提取颜色特征）
    ycbcr_img = np.require(rgb2ycbcr(img), dtype=np.float64)

    # 获取图像尺寸
    height, width, channels = img.shape

    # 窗口中心偏移量（窗口一半的整数部分）
    offset = np.floor((winsize - 1) / 2).astype(int)

    # 计算滑动窗口的起始坐标范围
    rangex = range(0, width - winsize + 1, stepsize)
    rangey = range(0, height - winsize + 1, stepsize)

    # 初始化特征矩阵（4维特征：Cb均值、Cr均值、y坐标、x坐标）
    num_features = len(rangex) * len(rangey)
    X = np.zeros((4, num_features), dtype=np.float64)

    # 索引计数器
    feature_idx = 0

    # 用于转换为Matlab索引（1-based）的偏移量
    index_offset = 1 if follow_matlab else 0

    # 滑动窗口提取特征
    for x in rangex:
        for y in rangey:
            # 提取当前窗口的Cb和Cr通道
            cb_window = ycbcr_img[y:y + winsize, x:x + winsize, 1]
            cr_window = ycbcr_img[y:y + winsize, x:x + winsize, 2]

            # 计算特征：Cb均值、Cr均值、窗口中心y坐标、窗口中心x坐标
            X[:, feature_idx] = [
                np.mean(cb_window.flatten()),
                np.mean(cr_window.flatten()),
                y + offset + index_offset,
                x + offset + index_offset
            ]

            feature_idx += 1

    # 存储位置信息的字典
    location_info = {
        'rangex': rangex,
        'rangey': rangey,
        'offset': offset,
        'width': width,
        'height': height,
        'stepsize': stepsize,
        'winsize': winsize,
        'follow_matlab': follow_matlab
    }

    return X, location_info


def labels2seg(labels, location_info):
    """
    将聚类标签转换为分割图像

    参数:
        labels: 每个特征点的聚类标签
        location_info: 从getfeatures获取的位置信息字典

    返回:
        np.ndarray: 与原图尺寸一致的分割图像
    """
    # 初始化分割图像（与原图尺寸相同）
    segm = np.zeros((location_info['height'], location_info['width']), dtype=np.int32)

    # 计算步长相关的填充范围
    step = location_info['stepsize']
    rstep = np.int32(np.floor(step / 2.0))
    stepbox = range(-rstep, step - rstep)

    # 获取所有窗口中心的坐标
    rx = np.asarray(location_info['rangex'], dtype=np.int32) + location_info['offset']
    ry = np.asarray(location_info['rangey'], dtype=np.int32) + location_info['offset']

    # 将标签重塑为与窗口网格对应的形状（按列优先，与Matlab一致）
    labels_reshaped = labels.reshape((ry.size, rx.size), order='F')

    # 填充分割图像：每个窗口中心周围的区域赋相同标签
    for i in stepbox:
        for j in stepbox:
            # 计算当前偏移对应的坐标，并确保不越界
            y_coords = ry + j
            x_coords = rx + i
            valid_y = (y_coords >= 0) & (y_coords < location_info['height'])
            valid_x = (x_coords >= 0) & (x_coords < location_info['width'])

            if np.any(valid_y) and np.any(valid_x):
                segm[np.ix_(y_coords[valid_y], x_coords[valid_x])] = labels_reshaped[valid_y, :][:, valid_x]

    # 处理边界区域（如果有未填充的像素）
    min_x = np.min(rx) + stepbox[0] - 1
    max_x = np.max(rx) + stepbox[-1] + 1
    min_y = np.min(ry) + stepbox[0] - 1
    max_y = np.max(ry) + stepbox[-1] + 1

    # 填充左边界
    if 0 <= min_x:
        segm[:, 0:min_x + 1] = segm[:, min_x + 1].reshape(-1, 1)

    # 填充右边界
    if max_x < location_info['width']:
        segm[:, max_x:] = segm[:, max_x - 1].reshape(-1, 1)

    # 填充上边界
    if 0 < min_y:
        segm[0:min_y + 1, :] = segm[min_y + 1, :].reshape(1, -1)

    # 填充下边界
    if max_y < location_info['height']:
        segm[max_y:, :] = segm[max_y - 1, :].reshape(1, -1)

    return segm


def demo():
    """演示图像分割流程：特征提取->聚类->生成分割图->着色显示"""
    # 加载图像（确保图像路径正确）
    try:
        img = Image.open('./PA2-cluster-images/images/12003.jpg')
    except FileNotFoundError:
        raise FileNotFoundError("请确保图像路径正确，当前路径: 'images/12003.jpg'")

    # 显示原图
    pl.subplot(1, 3, 1)
    pl.imshow(img)
    pl.title('original_img')
    pl.axis('off')

    # 提取图像特征（步长为7）
    X, location_info = getfeatures(img, 7)

    # 使用K-means聚类（2个聚类中心）
    # 先对特征进行白化处理，再聚类
    whitened_X = vq.whiten(X.T)  # 转置为每行一个特征
    centroids, labels = vq.kmeans2(whitened_X, 2, iter=1000, minit='random')
    labels = labels + 1  # 标签从1开始（与Matlab习惯一致）

    # 从聚类标签生成分割图像
    segm = labels2seg(labels, location_info)

    # 显示分割图
    pl.subplot(1, 3, 2)
    pl.imshow(segm, cmap='tab10')
    pl.title('splited results')
    pl.axis('off')

    # 对分割结果进行着色
    colored_segm = colorsegms(segm, img)

    # 显示着色后的分割图
    pl.subplot(1, 3, 3)
    pl.imshow(colored_segm)
    pl.title('colored results')
    pl.axis('off')

    # 显示所有图像
    pl.tight_layout()
    pl.show()


def main():
    """主函数，调用演示函数"""
    demo()


if __name__ == '__main__':
    main()