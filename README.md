# QuantumChat
[![Python application test with Github Actions](https://github.com/LeninKatta45/QuantumChat/actions/workflows/main.yml/badge.svg)](https://github.com/LeninKatta45/QuantumChat/actions/workflows/main.yml)
Quantum Chat
Overview
Quantum Chat is an AI-powered chatbot designed to provide accurate and real-time equity research responses. It leverages state-of-the-art models like Retrieval-Augmented Generation (RAG) and integrates advanced language models (like Google Gemini LLM) to offer contextual and precise answers for financial analysis and decision-making.

Features
Natural Language Processing (NLP): Utilizes advanced NLP techniques for understanding and generating human-like responses.
Equity Research Integration: Provides accurate and summarized responses for equity research queries.
Scalable Architecture: Built with scalable microservices using FastAPI and Flask, deployable on AWS.
Efficient Retrieval System: Uses LangChain and FAISS for fast and efficient information retrieval.
Customizability: Easily adaptable to other domains beyond equity research.
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/LeninKatta45/quantum-chat.git
cd quantum-chat
Set up a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Set up environment variables for API keys and other configurations in a .env file:

env
Copy code
OPENAI_API_KEY=your_openai_api_key
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
Usage
Start the server:

bash
Copy code
uvicorn app.main:app --reload
Access the Quantum Chat API at http://localhost:8000/docs for API documentation and testing.

Use the provided API endpoints to interact with the chatbot for equity research or any other domain-specific tasks.

Technologies Used
Programming Languages: Python, JavaScript
Frameworks and Libraries: FastAPI, Flask, LangChain, FAISS, Google Gemini LLM, Transformers
Tools and Platforms: Docker, AWS, GitHub Actions, MLflow, TensorFlow, PyTorch
Database: MongoDB, Vector Database
Machine Learning Techniques: RAG, Transformer-based models, NLP
Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Make your changes.
Commit your changes (git commit -m 'Add new feature').
Push to the branch (git push origin feature-branch).
Open a pull request.
License
This project is licensed under the MIT License. See the LICENSE file for more details.

Contact
For any queries or feedback, please reach out:

Lenin Balaji Katta
leninbalaji45@gmail.com
www.linkedin.com/in/leninkatta
