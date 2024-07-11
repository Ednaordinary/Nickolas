from dotenv import load_dotenv
import nextcord as discord
import subprocess
import asyncio
import time
import cv2
import os

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
intents = discord.Intents.all()
client = discord.Client(intents=intents)

scene_queue = []

def scene_runner():
    while True:
        if scene_queue != []:
            current_scene = scene_queue[0]
            try:
                capture = cv2.VideoCapture(current_scene.path + "video.mp4")
            except:
                asyncio.run_coroutine_threadsafe(coro=current_scene.interaction.edit_original_message("Failed to load video."), loop=client.loop)
            else:
                frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if frames < 100:
                    asyncio.run_coroutine_threadsafe(
                        coro=current_scene.interaction.edit_original_message("Video only has " + str(frames) + " frames! Please use a video with at least 100 frames."), loop=client.loop)
                else:
                    os.mkdir(current_scene.path + "frames")
                    fps_in = capture.get(cv2.CAP_PROP_FPS)
                    fps_out = int(80 / frames * fps_in)
                    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    if height > width:
                        scaleargs = "-vf scale=-2:1280:flags=lanczos,setsar=1:1"
                    else:
                        scaleargs = "-vf scale=1280:-2:flags=lanczos,setsar=1:1"
                    call_code = subprocess.check_call("ffmpeg -i " + current_scene.path + "video.mp4 " + scaleargs + " -r " + str(fps_out) + " " + current_scene.path + "frames/%d.png", shell=True)
                    if call_code != 0:
                        asyncio.run_coroutine_threadsafe(
                            coro=current_scene.interaction.edit_original_message(
                                "Something went wrong while processing the video"), loop=client.loop)
                        continue
                    os.makedirs(current_scene.path + "distorted/sparce", exist_ok=True)
                    call_code = subprocess.check_call("colmap feature_extractor --database_path " + current_scene.path + "distorted/database.db --image_path " + current_scene.path + "frames --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV --SiftExtraction.use_gpu True", shell=True)
                    if call_code != 0:
                        asyncio.run_coroutine_threadsafe(
                            coro=current_scene.interaction.edit_original_message(
                                "Something went wrong while extracting features."), loop=client.loop)
                        continue
                    call_code = subprocess.check_call("colmap exhaustive_matcher --database_path " + current_scene.path + "distorted/database.db --SiftMatching.use_gpu True", shell=True)
                    if call_code != 0:
                        asyncio.run_coroutine_threadsafe(
                            coro=current_scene.interaction.edit_original_message(
                                "Something went wrong while matching features. Try a video with more frame overlap!"), loop=client.loop)
                        continue
                    call_code = subprocess.check_call("colmap mapper --database_path " + current_scene.path + "distorted/database.db --SiftMatching.use_gpu True --image_path " + current_scene.path + "frames --output_path" + current_scene.path + "distorted/sparse --Mapper.ba_global_function_tolerance=0.000001")
                    if call_code != 0:
                        asyncio.run_coroutine_threadsafe(
                            coro=current_scene.interaction.edit_original_message(
                                "Something went wrong while mapping frames. Try a video with more frame overlap!"), loop=client.loop)
                        continue
                    distorted = os.listdir(current_scene.path + "sparse")
                    os.makedirs(current_scene.path + "sparse/0", exist_ok=True)
                    for image in distorted:

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
        await interaction.response.send_message("Please upload an mp4. Your current attachment is " + str(video.content_type))
    else:
        await interaction.response.send_message("Adding to the queue...")
        await video.save("videos/" + str(interaction.message.id) + "/video.mp4")
        global scene_queue
        scene_queue.append(SceneRequest(path="videos/" + str(interaction.message.id) + "/", interaction=interaction))

client.run(TOKEN)