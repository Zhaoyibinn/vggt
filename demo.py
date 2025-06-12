import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import cv2
import matplotlib.pylab as plt
import numpy as np

def connect_images_and_points(image_paths, point_tensor):
    """
    拼接图像并连接对应点
    
    参数:
    image_paths: 三张图像的路径列表
    point_tensor: 形状为[1, 3, 2, 2]的点坐标张量，格式为[批次, 图像索引, 点索引, (x, y)]
    """
    # 1. 读取图像
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"无法读取图像: {path}")
        images.append(img)
    
    # 2. 横向拼接图像
    combined_image = cv2.hconcat(images)
    
    # 3. 处理点坐标（假设tensor已转换为像素坐标，此处以示例数据演示）
    # 示例：假设point_tensor是numpy数组，形状为(1, 3, 2, 2)
    if isinstance(point_tensor, list):
        point_tensor = np.array(point_tensor)
    
    # 检查张量形状
    # if point_tensor.shape != (1, 3, 2, 2):
    #     raise ValueError(f"点张量形状应为(1, 3, 2, 2)，当前形状为{point_tensor.shape}")
    
    # 4. 计算每张图在拼接图中的起始位置
    img_widths = [img.shape[1] for img in images]
    start_positions = [0]
    for i in range(1, len(img_widths)):
        start_positions.append(start_positions[i-1] + img_widths[i-1])
    
    # 5. 在拼接图上绘制连线
    result_image = combined_image.copy()
    color = (0, 255, 0)  # 绿色(BGR)
    thickness = 2
    
    # 绘制第一条线（连接每个图的第一个点）
    points_line1 = []
    for i in range(3):
        x = point_tensor[0, i, 0, 0] + start_positions[i]
        y = point_tensor[0, i, 0, 1]
        points_line1.append((int(x), int(y)))
    
    # 绘制第二条线（连接每个图的第二个点）
    points_line2 = []
    # for i in range(3):
    #     x = point_tensor[0, i, 1, 0] + start_positions[i]
    #     y = point_tensor[0, i, 1, 1]
    #     points_line2.append((int(x), int(y)))
    
    # 绘制连线（使用线段连接相邻点）
    for i in range(len(points_line1) - 1):
        cv2.line(result_image, points_line1[i], points_line1[i+1], color, thickness)
        # cv2.line(result_image, points_line2[i], points_line2[i+1], color, thickness)
    
    # 6. 显示结果（可选：保存图像）
    plt.figure(figsize=(12, 6))
    # OpenCV是BGR格式，转换为RGB显示
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title("line")
    plt.axis("off")
    plt.savefig("test.png")
    
    return result_image

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
# model = VGGT.from_pretrained("weight/model.pt").to(device)
model = VGGT().to(device)
pretrained_dict = torch.load('weight/model.pt')
# model_dict = model.state_dict()
model.load_state_dict(pretrained_dict)

# Load and preprocess example images (replace with your own image paths)
image_names = ["examples/kitchen/images/00.png", "examples/kitchen/images/01.png", "examples/kitchen/images/02.png"]  
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)
        


from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        images = images[None]  # add batch dimension
        aggregated_tokens_list, ps_idx = model.aggregator(images)
                
    # Predict Cameras
    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
    # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    # Predict Depth Maps
    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    # Predict Point Maps
    point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
        
    # Construct 3D Points from Depth Maps and Cameras
    # which usually leads to more accurate 3D points than point map branch
    point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                extrinsic.squeeze(0), 
                                                                intrinsic.squeeze(0))

    # Predict Tracks
    # choose your own points to track, with shape (N, 2) for one scene
    query_points = torch.FloatTensor([[60.72, 259.94]]).to(device)
    track_list, vis_score, conf_score = model.track_head(aggregated_tokens_list, images, ps_idx, query_points=query_points[None])
    connect_images_and_points(image_names ,track_list[0])