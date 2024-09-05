# QuantumChat
[![Python application test with Github Actions](https://github.com/LeninKatta45/QuantumChat/actions/workflows/main.yml/badge.svg)](https://github.com/LeninKatta45/QuantumChat/actions/workflows/main.yml)

## Overview

**Quantum Chat** is an AI-powered chatbot designed to provide accurate and real-time equity research responses. It leverages state-of-the-art models like **Retrieval-Augmented Generation (RAG)** and integrates advanced language models (like Google Gemini LLM) to offer contextual and precise answers for financial analysis and decision-making.

## Features

- **Natural Language Processing (NLP):** Utilizes advanced NLP techniques for understanding and generating human-like responses.
- **Equity Research Integration:** Provides accurate and summarized responses for equity research queries.
- **Scalable Architecture:** Built with scalable microservices using **FastAPI** and **Flask**, deployable on AWS.
- **Efficient Retrieval System:** Uses **LangChain** and **FAISS** for fast and efficient information retrieval.
- **Customizability:** Easily adaptable to other domains beyond equity research.
  
## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/LeninKatta45/quantum-chat.git
    cd quantum-chat
    ```

2. Set up a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Set up environment variables for API keys and other configurations in a `.env` file:

    ```env
    OPENAI_API_KEY=your_openai_api_key
    AWS_ACCESS_KEY_ID=your_aws_access_key
    AWS_SECRET_ACCESS_KEY=your_aws_secret_key
    ```

## Usage

1. Start the server:

    ```bash
    uvicorn app.main:app --reload
    ```

2. Access the Quantum Chat API at `http://localhost:8000/docs` for API documentation and testing.

3. Use the provided API endpoints to interact with the chatbot for equity research or any other domain-specific tasks.

## Technologies Used

- **Programming Languages:** Python, JavaScript
- **Frameworks and Libraries:** FastAPI, Flask, LangChain, FAISS, Google Gemini LLM, Transformers
- **Tools and Platforms:** Docker, AWS, GitHub Actions, MLflow, TensorFlow, PyTorch
- **Database:** MongoDB, Vector Database
- **Machine Learning Techniques:** RAG, Transformer-based models, NLP

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For any queries or feedback, please reach out:

- **Lenin Balaji Katta**
- [Email](mailto:leninbalaji45@gmail.com)
- [LinkedIn](https://www.linkedin.com/in/leninkatta)
- [GitHub](https://github.com/LeninKatta45)

