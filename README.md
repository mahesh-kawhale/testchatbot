# testchatbot

# 💬 TestChatbot

A full-stack AI-powered chatbot built with **React** (frontend) and **Flask** (backend).  
The bot can handle conversations, fetch responses from a model or API, and provide an interactive chat interface similar to ChatGPT.

---

## 🚀 Features
- **Modern UI** inspired by ChatGPT
- **Flask REST API backend** for processing chat requests
- **React frontend** with responsive design
- **Real-time messaging**
- **Environment-based configuration**
- **Easily deployable** to platforms like Vercel (frontend) and Render/Heroku (backend)

---

## 🛠 Tech Stack
**Frontend**
- React
- Axios
- Custom CSS

**Backend**
- Flask
- Flask-CORS
- Python 3.9+

---

## 📂 Project Structure
testchatbot/
├── backend/
│ ├── app.py # Flask API entry point
│ ├── requirements.txt
│ └── ...
├── frontend/
│ ├── src/
│ │ ├── components/ # UI components
│ │ ├── pages/ # Pages
│ │ └── App.js
│ ├── package.json
│ └── ...
└── README.md



---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/mahesh-kawhale/testchatbot.git
cd testchatbot


Backend Setup (Flask)

cd backend
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
Backend will run on http://127.0.0.1:5000

3️⃣ Frontend Setup (React)

cd frontend
npm install
npm start
Frontend will run on http://localhost:3000

🔄 Environment Variables
Backend (.env):


OPENAI_API_KEY=your_api_key_here
Frontend (.env):


REACT_APP_API_URL=http://127.0.0.1:5000