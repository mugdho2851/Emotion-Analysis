{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fbf4eeec-fe8e-4a0a-8c8a-f30c6424d6af",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Quantum computing! It's a fascinating topic, and I'd be happy to explain it in simple terms.\n",
      "\n",
      "**Classical Computing**\n",
      "\n",
      "First, let's start with classical computing, which is what we use today. Imagine you have a big library with millions of books. Each book represents a piece of information, like a number or a word. To process this information, a classical computer uses \"bits\" to store and manipulate the data. A bit is either a 0 or a 1, like a light switch that's either on (1) or off (0).\n",
      "\n",
      "**Quantum Computing**\n",
      "\n",
      "Now, imagine a magical library where each book can exist in multiple places at the same time! This is kind of like a quantum computer. Instead of using bits, quantum computers use \"qubits\" (quantum bits). Qubits are special because they can exist in multiple states simultaneously, like being both 0 and 1 at the same time. This property is called superposition.\n",
      "\n",
      "Think\n"
     ]
    },
    {
     "name": "stdin",
     "output_type": "stream",
     "text": [
      "You:  what time is it now?\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Bot: I'm just an AI, I don't have access to real-time information, so I don't know the current time. However, I can suggest some ways for you to find out the current time:\n",
      "\n",
      "1. Check your phone or watch: Most phones and watches display the current time.\n",
      "2. Look at a clock: If you're near a clock, you can check the time on it.\n",
      "3. Use an online clock: You can search for \"current time\" or \"what time is it\" on a search engine like Google, and it will show you the current time.\n",
      "4. Ask a voice assistant: If you have a voice assistant like Alexa, Google Assistant, or Siri, you can ask it \"what time is it?\"\n",
      "\n",
      "I hope these suggestions help!\n"
     ]
    },
    {
     "name": "stdin",
     "output_type": "stream",
     "text": [
      "You:  hello\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Bot: Hello! It's nice to meet you. Is there something I can help you with, or would you like to chat?\n"
     ]
    }
   ],
   "source": [
    "import requests\n",
    "import json\n",
    "\n",
    "DEEPINFRA_API_KEY = \"HW1IAKbRFSFyqmUVIuA9WO5EnvsLeKfD\"\n",
    "\n",
    "MODEL_URL = \"https://api.deepinfra.com/v1/openai/chat/completions\"\n",
    "\n",
    "headers = {\n",
    "    \"Authorization\": f\"Bearer {DEEPINFRA_API_KEY}\",\n",
    "    \"Content-Type\": \"application/json\",\n",
    "}\n",
    "\n",
    "def chat_with_llama(prompt):\n",
    "    data = {\n",
    "        \"model\": \"meta-llama/Llama-2-70b-chat-hf\",\n",
    "        \"messages\": [{\"role\": \"user\", \"content\": prompt}],\n",
    "        \"max_tokens\": 200, \n",
    "    }\n",
    "    \n",
    "    response = requests.post(MODEL_URL, headers=headers, json=data)\n",
    "    return response.json()\n",
    "\n",
    "\n",
    "user_input = \"Explain quantum computing in simple terms.\"\n",
    "response = chat_with_llama(user_input)\n",
    "\n",
    "\n",
    "print(response[\"choices\"][0][\"message\"][\"content\"])\n",
    "\n",
    "while True:\n",
    "    user_input = input(\"You: \")\n",
    "    if user_input.lower() in [\"exit\", \"quit\"]:\n",
    "        break\n",
    "    response = chat_with_llama(user_input)\n",
    "    print(\"Bot:\", response[\"choices\"][0][\"message\"][\"content\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "57e4cd59-e5d5-4bcb-8765-04a325dcd2f9",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
