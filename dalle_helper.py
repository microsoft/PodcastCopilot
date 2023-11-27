# Helper class for DALL-E
# The following class creates a simple wrapper on the Azure OpenAI REST endpoints. It will simplify the steps for calling the text-to-image API to submit your request and then poll for the results

import requests
import time

class ImageClient:
    def __init__(self, endpoint, key, deployment_name = "Dalle3", api_version = "2023-12-01-preview", verbose=False):
        # These are the paramters for the class:
        # ### endpoint: The endpoint for your Azure OpenAI resource
        # ### key: The API key for your Azure OpenAI resource
        # ### deployment_name: The deployment name for Dall-E
        # ### api_version: The API version to use. This is optional and defaults to the latest version
        self.endpoint = endpoint
        self.api_key = key
        self.api_version = api_version
        self.deployment_name = deployment_name
        self.verbose = verbose

    def text_to_image(self, prompt):
        # this method makes the text-to-image API call. It will return the raw response from the API call
        reqURL = requests.models.PreparedRequest()
        params = {'api-version':self.api_version}
        #the full endpoint will look something like this https://YOUR_AOAI_RESOURCE_NAME.openai.azure.com/openai/deployments/<deployment-name>/images/generations
        reqURL.prepare_url(self.endpoint + f"openai/deployments/{self.deployment_name}/images/generations", params) 
        if self.verbose:
            print("Sending a POST call to the following URL: {URL}".format(URL=reqURL.url))

        #Construct the data payload for the call. This includes the prompt text as well as many optional parameters.
        payload = {"prompt": prompt}
        r = requests.post(reqURL.url, 
            headers={
                "Api-key": self.api_key,
                "Content-Type": "application/json"
            },
            json = payload
        )

        # Response Body example:
        #  {
        #   "created": 1698342300,
        #   "data": [
        #       {
        #           "url": "https://dalletipusw2.blob.core.windows.net/private/images/e5451cc6-b1ad-4747-bd46-b89a3a3b8bc3/generated_00.png?se=2023-10-27T17%3A45%3A09Z&..."
        #       },
        #       {
        #           "url": "https://dalletipusw2.blob.core.windows.net/private/images/e5451cc6-b1ad-4747-bd46-b89a3a3b8bc3/generated_01.png?se=2023-10-27T17%3A45%3A09Z&..."
        #       }],
        #   "revised_prompt": "A vivid, natural representation of Microsoft Clippy wearing a cowboy hat."
        #   }

        if r.status_code != 200:
            print("Error: {error}".format(error=r.json()))

        data = r.json()
        if self.verbose:
            print('text-to-image API response body:')
            print(data)
        return r


    def getImage(self, contentUrl):
        # Download the images from the given URL
        r = requests.get(contentUrl)
        return r


    def generateImage(self, prompt):
        submission = self.text_to_image(prompt)
        if self.verbose:
            print('Response code from submission')
            print(submission.status_code)
            print('Response body:')
            print(submission.json())
        if submission.status_code == 200:
            contentUrl = submission.json()['data'][0]['url']
        else:
            print('Not a 200 response')
            return "-1"
        
        image = self.getImage(contentUrl)
        return contentUrl, image.content



