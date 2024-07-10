from dotenv import load_dotenv
import nextcord as discord
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
            capture = cv2.VideoCapture("")
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
        await video.save("videos/" + str(interaction.message.id) + ".mp4")
        global scene_queue
        scene_queue.append(SceneRequest(path="videos/" + str(interaction.message.id) + ".mp4", interaction=interaction))

client.run(TOKEN)