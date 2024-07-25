import threading
from argparse import ArgumentParser
import torch.nn.functional as F
import tqdm
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
import vram

from arguments import ParamGroup
from gaussian_model import GaussianModel
from scene import Scene
from render import render
from arguments import OptimizationParams, PipelineParams
from loss import l1_loss, ssim, _ssim

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
intents = discord.Intents.all()
client = discord.Client(intents=intents)

scene_queue = []

# Almost all of this comes from https://github.com/graphdeco-inria/gaussian-splatting/
# please go check it out!

async def async_scene_runner():
    global scene_queue
    while True:
        if scene_queue != []:
            current_scene = scene_queue[0]
            print(current_scene.path, current_scene.interaction, current_scene.message)
            try:
                capture = cv2.VideoCapture(current_scene.path + "video.mp4")
            except:
                asyncio.run_coroutine_threadsafe(
                    coro=current_scene.message.edit("Failed to load video."), loop=client.loop)
            else:
                frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if frames < 100:
                    asyncio.run_coroutine_threadsafe(
                        coro=current_scene.message.edit(
                            "Video only has " + str(frames) + " frames! Please use a video with at least 100 frames."),
                        loop=client.loop)
                else:
                    os.makedirs(current_scene.path + "frames", exist_ok=True)
                    fps_in = capture.get(cv2.CAP_PROP_FPS)
                    fps_out = int(100 / frames * fps_in)
                    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    if height > width:
                        scaleargs = "-vf scale=-2:1280:flags=lanczos,setsar=1:1"
                    else:
                        scaleargs = "-vf scale=1280:-2:flags=lanczos,setsar=1:1"
                    try:
                        call_code = subprocess.check_call(
                            "ffmpeg -i " + current_scene.path + "video.mp4 " + scaleargs + " -r " + str(
                                fps_out) + " " + current_scene.path + "frames/%d.png", shell=True)
                    except: call_code = 1
                    if call_code != 0:
                        asyncio.run_coroutine_threadsafe(
                            coro=current_scene.message.edit(
                                "Something went wrong while processing the video"), loop=client.loop)
                        scene_queue.pop(0)
                        continue
                    vram.allocate("Nickolas")
                    async for i in vram.wait_for_allocation("Nickolas"):
                        asyncio.run_coroutine_threadsafe(
                            coro=current_scene.message.edit(
                                "Waiting for " + str(i) + " before loading model"), loop=client.loop)
                        pass
                    os.makedirs(current_scene.path + "distorted/sparse", exist_ok=True)
                    try:
                        call_code = subprocess.check_call(
                            "colmap feature_extractor --database_path " + current_scene.path + "distorted/database.db --image_path " + current_scene.path + "frames --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV --SiftExtraction.use_gpu True",
                            shell=True)
                    except: call_code = 1
                    if call_code != 0:
                        asyncio.run_coroutine_threadsafe(
                            coro=current_scene.message.edit(
                                "Something went wrong while extracting features."), loop=client.loop)
                        scene_queue.pop(0)
                        continue
                    asyncio.run_coroutine_threadsafe(
                        coro=current_scene.message.edit(
                            "Finished extracting features"), loop=client.loop)
                    try:
                        call_code = subprocess.check_call(
                            "colmap exhaustive_matcher --database_path " + current_scene.path + "distorted/database.db --SiftMatching.use_gpu True",
                            shell=True)
                    except: call_code = 1
                    if call_code != 0:
                        asyncio.run_coroutine_threadsafe(
                            coro=current_scene.message.edit(
                                "Something went wrong while matching features. Try a video with more frame overlap!"),
                            loop=client.loop)
                        scene_queue.pop(0)
                        continue
                    asyncio.run_coroutine_threadsafe(
                        coro=current_scene.message.edit(
                            "Finished matching features"),
                        loop=client.loop)
                    vram.deallocate("Nickolas")
                    limiter = time.time()
                    try:
                        process = subprocess.Popen("colmap mapper --database_path " + current_scene.path + "distorted/database.db --image_path " + current_scene.path + "frames --output_path " + current_scene.path + "distorted/sparse --Mapper.ba_global_function_tolerance=0.000001", shell=True, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True, text=True)
                        for line in process.stderr:
                            if "Registering image" in line:
                                try:
                                    percent = line.split("(")[-1][:-2]
                                except:
                                    pass
                                else:
                                    if time.time() >  limiter + 0.8:
                                        asyncio.run_coroutine_threadsafe(
                                            coro=current_scene.message.edit(
                                                "Mapper: " + str(percent) + "%"),
                                            loop=client.loop)
                                        limiter = time.time()
                    except: call_code = 1
                    else:
                        call_code = 0
                    if call_code != 0:
                        asyncio.run_coroutine_threadsafe(
                            coro=current_scene.message.edit(
                                "Something went wrong while mapping frames. Try a video with more frame overlap!"),
                            loop=client.loop)
                        scene_queue.pop(0)
                        continue
                    asyncio.run_coroutine_threadsafe(
                        coro=current_scene.message.edit(
                            "Finished mapping frames"),
                        loop=client.loop)
                    try:
                        call_code = subprocess.check_call(
                            "colmap image_undistorter --image_path " + current_scene.path + "frames --output_path " + current_scene.path + " --input_path " + current_scene.path + "distorted/sparse/0 --output_path " + current_scene.path + " --output_type COLMAP", shell=True)
                    except Exception as e:
                        print(repr(e))
                        call_code = 1
                    if call_code != 0:
                        asyncio.run_coroutine_threadsafe(
                            coro=current_scene.message.edit(
                                "Something went wrong while undistorting frames."),
                            loop=client.loop)
                        scene_queue.pop(0)
                        continue
                    distorted = os.listdir(current_scene.path + "/distorted/sparse")
                    os.makedirs(current_scene.path + "distorted/sparse/0", exist_ok=True)
                    for image in distorted:
                        if image == '0':
                            continue
                        source_file = os.path.join(current_scene.path, "sparse", image)
                        dest_file = os.path.join(current_scene.path, "sparse", "0", image)
                        shutil.move(source_file, dest_file)
                    vram.allocate("Nickolas")
                    async for i in vram.wait_for_allocation("Nickolas"):
                        asyncio.run_coroutine_threadsafe(
                            coro=current_scene.message.edit(
                                "Waiting for " + str(i) + " before loading model"), loop=client.loop)
                        pass
                    first_iter = 0
                    gaussians = GaussianModel(3)
                    pipe = PipelineParams()
                    scene = Scene(current_scene.path + "model", current_scene.path, gaussians)
                    gaussians.training_setup(OptimizationParams())
                    bg_color = [0, 0, 0]
                    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                    viewpoint_stack = None
                    ema_loss_for_log = 0.0
                    progress_bar = tqdm(range(first_iter, 30000), desc="Training progress")
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
                        gt_image = viewpoint_cam.original_image.cuda()
                        Ll1 = l1_loss(image, gt_image)
                        loss = 0.8 * Ll1 + 0.2 * (1.0 - ssim(image, gt_image))
                        loss.backward()
                        with torch.no_grad:
                            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                            progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                            progress_bar.update(1)
                            # Densification
                            if iteration < 15000:
                                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                                if iteration > 500 and iteration % 100 == 0:
                                    size_threshold = 20 if iteration > 3000 else None
                                    gaussians.densify_and_prune(0.0002, 0.005, scene.cameras_extent,
                                                                size_threshold)
                                if iteration % 3000 == 0:
                                    gaussians.reset_opacity()
                            # Optimizer
                            if iteration < 30000:
                                gaussians.optimizer.step()
                                gaussians.optimizer.zero_grad(set_to_none=True)
                            if iteration == 30000:
                                progress_bar.close()
                                scene.save("model")
                    vram.deallocate("Nickolas")
                    asyncio.run_coroutine_threadsafe(
                        coro=current_scene.message.edit(
                            "Done! (placeholder)"),
                        loop=client.loop)
                    scene_queue.pop(0)
        time.sleep(0.01)

def scene_runner():
    loop = asyncio.new_event_loop()
    loop.run_until_complete(async_scene_runner())


class SceneRequest:
    def __init__(self, path: str, interaction: discord.Interaction, message):
        self.path = path
        self.interaction = interaction
        self.message = message

@client.event
async def on_ready():
    print(f'{client.user.name} has connected to Discord!')

@client.slash_command(description="Make a 3D scene from a video")
async def scene(
        interaction: discord.Interaction,
        video: discord.Attachment,
):
    if not video.content_type == "video/mp4":
        await interaction.response.send_message(
            "Please upload an mp4. Your current attachment is " + str(video.content_type))
    else:
        message = await interaction.response.send_message("Adding to the queue...")
        if isinstance(message, discord.PartialInteractionMessage):
            message = await message.fetch()
            os.makedirs("videos/" + str(message.id))
        await video.save("videos/" + str(message.id) + "/video.mp4")
        global scene_queue
        scene_queue.append(SceneRequest(path="videos/" + str(message.id) + "/", interaction=interaction, message=message))


threading.Thread(target=scene_runner).start()
client.run(TOKEN)
