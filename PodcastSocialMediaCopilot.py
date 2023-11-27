# The Podcast Copilot will automatically create and post a LinkedIn promotional post for a new episode of the Behind the Tech podcast.  
# Given the audio recording of the episode, the copilot will use a locally-hosted Whisper model to transcribe the audio recording.
# The copilot uses the Dolly 2 model to extract the guest's name from the transcript.
# The copilot uses the Bing Search Grounding API to retrieve a bio for the guest.
# The copilot uses the GPT-4 model in the Azure OpenAI Service to generate a social media blurb for the episode, given the transcript and the guest's bio.
# The copilot uses the DALL-E 2 model to generate an image for the post.
# The copilot calls a LinkedIn plugin to post.

from pydub import AudioSegment
from pydub.silence import split_on_silence
import whisper
import torch
from langchain.chains import TransformChain, LLMChain, SequentialChain
from langchain.chat_models import AzureChatOpenAI
from langchain.llms import HuggingFacePipeline
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import requests
import time
from PIL import Image
from io import BytesIO
import datetime
import json
from dalle_helper import ImageClient

# For Dolly 2
from transformers import AutoTokenizer, TextStreamer
from optimum.onnxruntime import ORTModelForCausalLM
from instruct_pipeline import InstructionTextGenerationPipeline
import onnxruntime as ort
ort.set_default_logger_severity(3)

print("Imports are complete")


# Endpoint Settings
bing_search_url = "https://api.bing.microsoft.com/v7.0/search"
bing_subscription_key = "TODO"                              # Your key will look something like this: 00000000000000000000000000000000
openai_api_type = "azure"
openai_api_base = "https://TODO.openai.azure.com/"          # Your endpoint will look something like  this: https://YOUR_AOAI_RESOURCE_NAME.openai.azure.com/
openai_api_key = "TODO"                                     # Your key will look something like this: 00000000000000000000000000000000
gpt4_deployment_name = "gpt-4"
dalle_deployment_name = "Dalle3"

# We are assuming that you have all model deployments on the same Azure OpenAI service resource above.  If not, you can change these settings below to point to different resources.
gpt4_endpoint = openai_api_base                             # Your endpoint will look something like  this: https://YOUR_AOAI_RESOURCE_NAME.openai.azure.com/
gpt4_api_key = openai_api_key                               # Your key will look something like this: 00000000000000000000000000000000
dalle_endpoint = openai_api_base                            # Your endpoint will look something like  this: https://YOUR_AOAI_RESOURCE_NAME.openai.azure.com/
dalle_api_key = openai_api_key                              # Your key will look something like this: 00000000000000000000000000000000
plugin_model_url = openai_api_base
plugin_model_api_key = openai_api_key                       # Your key will look something like this: 00000000000000000000000000000000

# Inputs about the podcast
podcast_url = "https://www.microsoft.com/behind-the-tech"
podcast_audio_file = ".\PodcastSnippet.mp3"


# Step 1 - Call Whisper to transcribe audio
print("Calling Whisper to transcribe audio...\n")

# Chunk up the audio file 
sound_file = AudioSegment.from_mp3(podcast_audio_file)
audio_chunks = split_on_silence(sound_file, min_silence_len=1000, silence_thresh=-40 )
count = len(audio_chunks)
print("Audio split into " + str(count) + " audio chunks")

# Call Whisper to transcribe audio
model = whisper.load_model("base")
transcript = ""
for i, chunk in enumerate(audio_chunks):
    # If you have a long audio file, you can enable this to only run for a subset of chunks
    if i < 10 or i > count - 10:
        out_file = "chunk{0}.wav".format(i)
        print("Exporting", out_file)
        chunk.export(out_file, format="wav")
        result = model.transcribe(out_file)
        transcriptChunk = result["text"]
        print(transcriptChunk)
        
        # Append transcript in memory if you have sufficient memory
        transcript += " " + transcriptChunk

        # Alternatively, here's how to write the transcript to disk if you have memory constraints
        #textfile = open("chunk{0}.txt".format(i), "w")
        #textfile.write(transcript)
        #textfile.close()
        #print("Exported chunk{0}.txt".format(i))

print("Transcript: \n")
print(transcript)
print("\n")


# Step 2 - Make a call to a local Dolly 2.0 model optimized for Windows to extract the name of who I'm interviewing from the transcript
print("Calling a local Dolly 2.0 model optimized for Windows to extract the name of the podcast guest...\n")
repo_id = "microsoft/dolly-v2-7b-olive-optimized"
tokenizer = AutoTokenizer.from_pretrained(repo_id, padding_side="left")
model = ORTModelForCausalLM.from_pretrained(repo_id, provider="DmlExecutionProvider", use_cache=True, use_merged=True, use_io_binding=False)
streamer = TextStreamer(tokenizer, skip_prompt=True)
generate_text = InstructionTextGenerationPipeline(model=model, streamer=streamer, tokenizer=tokenizer, max_new_tokens=128, return_full_text=True, task="text-generation")
hf_pipeline = HuggingFacePipeline(pipeline=generate_text)
    
dolly2_prompt = PromptTemplate(
    input_variables=["transcript"],
    template="Extract the guest name on the Beyond the Tech podcast from the following transcript.  Beyond the Tech is hosted by Kevin Scott and Christina Warren, so they will never be the guests.  \n\n Transcript: {transcript}\n\n Host name: Kevin Scott\n\n Guest name: "
)

extract_llm_chain = LLMChain(llm=hf_pipeline, prompt=dolly2_prompt, output_key="guest")
guest = extract_llm_chain.predict(transcript=transcript)

print("Guest:\n")
print(guest)
print("\n")


# Step 3 - Make a call to the Bing Search Grounding API to retrieve a bio for the guest
def bing_grounding(input_dict:dict) -> dict:
    print("Calling Bing Search API to get bio for guest...\n")
    search_term = input_dict["guest"]
    print("Search term is " + search_term)

    headers = {"Ocp-Apim-Subscription-Key": bing_subscription_key}
    params = {"q": search_term, "textDecorations": True, "textFormat": "HTML"}
    response = requests.get(bing_search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    #print(search_results)

    # Parse out a bio.  
    bio = search_results["webPages"]["value"][0]["snippet"]
    
    print("Bio:\n")
    print(bio)
    print("\n")

    return {"bio": bio}

bing_chain = TransformChain(input_variables=["guest"], output_variables=["bio"], transform=bing_grounding)
bio = bing_chain.run(guest)


# Step 4 - Put bio in the prompt with the transcript
system_template="You are a helpful large language model that can create a LinkedIn promo blurb for episodes of the podcast Behind the Tech, when given transcripts of the podcasts.  The Behind the Tech podcast is hosted by Kevin Scott.\n"
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

user_prompt=PromptTemplate(
    template="Create a short summary of this podcast episode that would be appropriate to post on LinkedIn to promote the podcast episode.  The post should be from the first-person perspective of Kevin Scott, who hosts the podcast.\n" +
            "Here is the transcript of the podcast episode: {transcript} \n" +
            "Here is the bio of the guest: {bio} \n",
    input_variables=["transcript", "bio"],
)
human_message_prompt = HumanMessagePromptTemplate(prompt=user_prompt)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# Get formatted messages for the chat completion
blurb_messages = chat_prompt.format_prompt(transcript={transcript}, bio={bio}).to_messages()


# Step 5 - Make a call to Azure OpenAI Service to get a social media blurb, 
print("Calling GPT-4 model on Azure OpenAI Service to get a social media blurb...\n")
gpt4 = AzureChatOpenAI(
    openai_api_base=gpt4_endpoint,
    openai_api_version="2023-03-15-preview",
    deployment_name=gpt4_deployment_name,
    openai_api_key=gpt4_api_key,
    openai_api_type = openai_api_type,
)
#print(gpt4)   #shows parameters

output = gpt4(blurb_messages)
social_media_copy = output.content

gpt4_chain = LLMChain(llm=gpt4, prompt=chat_prompt, output_key="social_media_copy")

print("Social Media Copy:\n")
print(social_media_copy)
print("\n")


# Step 6 - Use GPT-4 to generate a DALL-E prompt
system_template="You are a helpful large language model that generates DALL-E prompts, that when given to the DALL-E model can generate beautiful high-quality images to use in social media posts about a podcast on technology.  Good DALL-E prompts will contain mention of related objects, and will not contain people or words.  Good DALL-E prompts should include a reference to podcasting along with items from the domain of the podcast guest.\n"
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

user_prompt=PromptTemplate(
    template="Create a DALL-E prompt to create an image to post along with this social media text: {social_media_copy}",
    input_variables=["social_media_copy"],
)
human_message_prompt = HumanMessagePromptTemplate(prompt=user_prompt)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# Get formatted messages for the chat completion
dalle_messages = chat_prompt.format_prompt(social_media_copy={social_media_copy}).to_messages()

# Call Azure OpenAI Service to get a DALL-E prompt 
print("Calling GPT-4 model on Azure OpenAI Service to get a DALL-E prompt...\n")
gpt4 = AzureChatOpenAI(
    openai_api_base=gpt4_endpoint,
    openai_api_version="2023-03-15-preview",
    deployment_name=gpt4_deployment_name,
    openai_api_key=gpt4_api_key,
    openai_api_type = openai_api_type,
)
#print(gpt4)   #shows parameters

output = gpt4(dalle_messages)
dalle_prompt = output.content

dalle_prompt_chain = LLMChain(llm=gpt4, prompt=chat_prompt, output_key="dalle_prompt")

print("DALL-E Prompt:\n")
print(dalle_prompt)
print("\n")


# For the demo, we showed the step by step execution of each chain above, but you can also run the entire chain in one step.
# You can uncomment and run the following code for an example.  Feel free to substitute your own transcript.
'''
transcript = "Hello, and welcome to Beyond the Tech podcast.  I am your host, Kevin Scott.  I am the CTO of Microsoft.  I am joined today by an amazing guest, Lionel Messi.  Messi is an accomplished soccer player for the Paris Saint-Germain football club.  Lionel, how are you doing today?"

podcast_copilot_chain = SequentialChain(
    chains=[extract_llm_chain, bing_chain, gpt4_chain, dalle_prompt_chain],
    input_variables=["transcript"],
    output_variables=["guest", "bio", "social_media_copy", "dalle_prompt"],
    verbose=True)
podcast_copilot = podcast_copilot_chain({"transcript":transcript})
print(podcast_copilot)		# This is helpful for debugging.  
social_media_copy = podcast_copilot["social_media_copy"]
dalle_prompt = podcast_copilot["dalle_prompt"]

print("Social Media Copy:\n")
print(social_media_copy)
print("\n")
'''


# Append "high-quality digital art" to the generated DALL-E prompt
dalle_prompt = dalle_prompt + ", high-quality digital art"


# Step 7 - Make a call to DALL-E model on the Azure OpenAI Service to generate an image 
print("Calling DALL-E model on Azure OpenAI Service to get an image for social media...\n")

# Establish the client class instance
client = ImageClient(dalle_endpoint, dalle_api_key, dalle_deployment_name, verbose=False) # change verbose to True for including debug print statements

# Generate an image
imageURL, postImage =  client.generateImage(dalle_prompt)
print("Image URL: " + imageURL + "\n")

# Write image to file - this is optional if you would like to have a local copy of the image
stream = BytesIO(postImage)
image = Image.open(stream).convert("RGB")
stream.close()
photo_path = ".\PostImage.jpg"
image.save(photo_path)
print("Image: saved to PostImage.jpg\n")


# Append the podcast URL to the generated social media copy
social_media_copy = social_media_copy + " " + podcast_url


# Step 8 - Call the LinkedIn Plugin for Copilots to do the post.
# Currently there is not support in the SDK for the plugin model on Azure OpenAI, so we are using the REST API directly.  
PROMPT_MESSAGES = [
    {
        "role": "system",
        "content": "You are a helpful large language model that can post a LinkedIn promo blurb for episodes of Behind the Tech with Kevin Scott, when given some text and a link to an image.\n",
    },
    {
        "role": "user",
        "content": 
            "Post the following social media text to LinkedIn to promote my latest podcast episode: \n" +
            "Here is the text to post: \n" + social_media_copy + "\n" +
            "Here is a link to the image that should be included with the post: \n" + imageURL + "\n",
    }, 
]

print("Calling GPT-4 model with plugin support on Azure OpenAI Service to post to LinkedIn...\n")

payload = {
    "messages": PROMPT_MESSAGES,
    "max_tokens": 1024,
    "temperature": 0.5,
    "n": 1,
    "stop": None
}

headers = {
    "Content-Type": "application/json",
    "api-key": plugin_model_api_key,
}

# Confirm whether it is okay to post, to follow Responsible AI best practices
print("The following will be posted to LinkedIn:\n")
print(social_media_copy + "\n")
confirm = input("Do you want to post this to LinkedIn? (y/n): ")
if confirm == "y":
    # Call a model with plugin support.
    response = requests.post(plugin_model_url, headers=headers, data=json.dumps(payload))
    
    #print (type(response))
    print("Response:\n")
    print(response)
    print("Headers:\n")
    print(response.headers)
    print("Json:\n")
    print(response.json())
    
    response_dict = response.json()
    print(response_dict["choices"][0]["messages"][-1]["content"])
    
# To use plugins, you must call a model that understands how to leverage them.  Support for plugins is in limited private preview
# for the Azure OpenAI service, and a LinkedIn plugin is coming soon!


