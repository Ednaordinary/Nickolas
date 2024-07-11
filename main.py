from dotenv import load_dotenv
import nextcord as discord
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
                capture = cv2.VideoCapture(current_scene.path)
            except:
                asyncio.run_coroutine_threadsafe(coro=current_scene.interaction.edit_original_message("Failed to load video."), loop=client.loop)
            else:
                frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if frames < 100:
                    asyncio.run_coroutine_threadsafe(
                        coro=current_scene.interaction.edit_original_message("Video only has " + str(frames) + " frames! Please use a video with at least 100 frames."), loop=client.loop)
                else:

        time.sleep(0.01)

class SceneRequest:
    def __init__(self, path, interaction):
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
        await video.save("videos/" + str(interaction.message.id) + ".mp4")
        global scene_queue
        scene_queue.append(SceneRequest(path="videos/" + str(interaction.message.id) + ".mp4", interaction=interaction))

client.run(TOKEN)