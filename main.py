from argparse import ArgumentParser
import torch.nn.functional as F
from dotenv import load_dotenv
import nextcord as discord
import subprocess
import asyncio
import random
import shutil
import torch
import time
import cv2
import os

from arguments import ParamGroup
from gaussian_model import GaussianModel
from scene import Scene
from render import render
from arguments import OptimizationParams, PipelineParams

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
intents = discord.Intents.all()
client = discord.Client(intents=intents)

scene_queue = []

# Almost all of this comes from https://github.com/graphdeco-inria/gaussian-splatting/
# please go check it out!


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def scene_runner():
    while True:
        if scene_queue != []:
            current_scene = scene_queue[0]
            try:
                capture = cv2.VideoCapture(current_scene.path + "video.mp4")
            except:
                asyncio.run_coroutine_threadsafe(
                    coro=current_scene.interaction.edit_original_message("Failed to load video."), loop=client.loop)
            else:
                frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if frames < 100:
                    asyncio.run_coroutine_threadsafe(
                        coro=current_scene.interaction.edit_original_message(
                            "Video only has " + str(frames) + " frames! Please use a video with at least 100 frames."),
                        loop=client.loop)
                else:
                    os.mkdir(current_scene.path + "frames")
                    fps_in = capture.get(cv2.CAP_PROP_FPS)
                    fps_out = int(100 / frames * fps_in)
                    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    if height > width:
                        scaleargs = "-vf scale=-2:1280:flags=lanczos,setsar=1:1"
                    else:
                        scaleargs = "-vf scale=1280:-2:flags=lanczos,setsar=1:1"
                    call_code = subprocess.check_call(
                        "ffmpeg -i " + current_scene.path + "video.mp4 " + scaleargs + " -r " + str(
                            fps_out) + " " + current_scene.path + "frames/%d.png", shell=True)
                    if call_code != 0:
                        asyncio.run_coroutine_threadsafe(
                            coro=current_scene.interaction.edit_original_message(
                                "Something went wrong while processing the video"), loop=client.loop)
                        continue
                    os.makedirs(current_scene.path + "distorted/sparce", exist_ok=True)
                    call_code = subprocess.check_call(
                        "colmap feature_extractor --database_path " + current_scene.path + "distorted/database.db --image_path " + current_scene.path + "frames --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV --SiftExtraction.use_gpu True",
                        shell=True)
                    if call_code != 0:
                        asyncio.run_coroutine_threadsafe(
                            coro=current_scene.interaction.edit_original_message(
                                "Something went wrong while extracting features."), loop=client.loop)
                        continue
                    asyncio.run_coroutine_threadsafe(
                        coro=current_scene.interaction.edit_original_message(
                            "Finished extracting features"), loop=client.loop)
                    call_code = subprocess.check_call(
                        "colmap exhaustive_matcher --database_path " + current_scene.path + "distorted/database.db --SiftMatching.use_gpu True",
                        shell=True)
                    if call_code != 0:
                        asyncio.run_coroutine_threadsafe(
                            coro=current_scene.interaction.edit_original_message(
                                "Something went wrong while matching features. Try a video with more frame overlap!"),
                            loop=client.loop)
                        continue
                    asyncio.run_coroutine_threadsafe(
                        coro=current_scene.interaction.edit_original_message(
                            "Finished matching features"),
                        loop=client.loop)
                    call_code = subprocess.check_call(
                        "colmap mapper --database_path " + current_scene.path + "distorted/database.db --SiftMatching.use_gpu True --image_path " + current_scene.path + "frames --output_path" + current_scene.path + "distorted/sparse --Mapper.ba_global_function_tolerance=0.000001")
                    if call_code != 0:
                        asyncio.run_coroutine_threadsafe(
                            coro=current_scene.interaction.edit_original_message(
                                "Something went wrong while mapping frames. Try a video with more frame overlap!"),
                            loop=client.loop)
                        continue
                    asyncio.run_coroutine_threadsafe(
                        coro=current_scene.interaction.edit_original_message(
                            "Finished mapping frames"),
                        loop=client.loop)
                    call_code = subprocess.check_call(
                        "colmap image_undistorter --image_path " + current_scene.path + "frames --output_path " + current_scene.path + " --input_path " + current_scene + "distorted/sparse/0")
                    distorted = os.listdir(current_scene.path + "sparse")
                    os.makedirs(current_scene.path + "sparse/0", exist_ok=True)
                    for image in distorted:
                        if image == '0':
                            continue
                        source_file = os.path.join(current_scene.path, "sparse", image)
                        dest_file = os.path.join(current_scene.path, "sparse", "0", image)
                        shutil.move(source_file, dest_file)
                    first_iter = 0
                    gaussians = GaussianModel(3)
                    pipe = PipelineParams()
                    scene = Scene(current_scene.path + "model", current_scene.path,gaussians)
                    gaussians.training_setup(OptimizationParams())
                    bg_color = [0, 0, 0]
                    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                    viewpoint_stack = None
                    ema_loss_for_log = 0.0
                    first_iter += 1
                    for iteration in range(first_iter, 30001):
                        gaussians.update_learning_rate(iteration)
                        if iteration % 1000 == 0:
                            gaussians.oneupSHdegree()
                        if not viewpoint_stack:
                            viewpoint_stack = scene.getTrainCameras().copy()
                        viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack)-1))
                        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                        image = render_pkg["render"]
                        viewspace_point_tensor = render_pkg["viewspace_points"]
                        visibility_filter = render_pkg["visibility_filter"]
                        radii = render_pkg["radii"]
                        gt_imag e=
        time.sleep(0.01)


class SceneRequest:
    def __init__(self, path: str, interaction: discord.Interaction):
        self.path = path
        self.interaction = interaction


@client.slash_command(description="Make a 3D scene from a video")
async def scene(
        interaction: discord.Interaction,
        video: discord.Attachment,
):
    if not video.content_type == "video/mp4":
        await interaction.response.send_message(
            "Please upload an mp4. Your current attachment is " + str(video.content_type))
    else:
        await interaction.response.send_message("Adding to the queue...")
        await video.save("videos/" + str(interaction.message.id) + "/video.mp4")
        global scene_queue
        scene_queue.append(SceneRequest(path="videos/" + str(interaction.message.id) + "/", interaction=interaction))


client.run(TOKEN)
