import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import ChatMessage from './components/ChatMessage';
// Removed SourceCitation import
import { FaPaperPlane, FaRobot, FaTimes, FaComments, FaMinus, FaPlus } from 'react-icons/fa';
import axios from 'axios';

// API configuration
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';
const API_KEY = process.env.REACT_APP_API_KEY || 'test-key';

function App() {
  const [messages, setMessages] = useState([
    { 
      role: 'assistant', 
      content: 'Hello! I\'m the Neutrino Tech Systems AI assistant. I provide concise answers about our services and technologies. How can I help you today?',
      sources: []
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [error, setError] = useState(null);
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Focus input when chat is opened
  useEffect(() => {
    if (isChatOpen && !isMinimized) {
      inputRef.current?.focus();
    }
  }, [isChatOpen, isMinimized]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!input.trim()) return;
    
    // Add user message
    const userMessage = { role: 'user', content: input, sources: [] };
    setMessages(prev => [...prev, userMessage]);
    
    // Clear input and show loading
    setInput('');
    setIsLoading(true);
    setError(null);
    
    try {
      // Add temporary assistant message for streaming
      const tempId = Date.now().toString();
      setMessages(prev => [...prev, { 
        id: tempId,
        role: 'assistant', 
        content: '', 
        sources: [],
        isStreaming: true 
      }]);
      
      // Use streaming endpoint
      const response = await fetch(`${API_URL}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': API_KEY
        },
        body: JSON.stringify({
          message: userMessage.content,
          session_id: sessionId
        })
      });
      
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let sources = [];
      
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.substring(6));
              
              if (data.type === 'token') {
                // Update the streaming message content with each token
                setMessages(prev => prev.map(msg =>
                  msg.id === tempId
                    ? { ...msg, content: data.content || '' }
                    : msg
                ));
              } else if (data.type === 'sources') {
                sources = data.sources;
              } else if (data.type === 'done') {
                // Update the message with sources when done
                setMessages(prev => prev.map(msg =>
                  msg.id === tempId
                    ? { ...msg, sources, isStreaming: false }
                    : msg
                ));
              } else if (data.type === 'error') {
                setError(data.error);
                // Remove the streaming message
                setMessages(prev => prev.filter(msg => msg.id !== tempId));
              } else {
                // Update the streaming message content
                setMessages(prev => prev.map(msg => 
                  msg.id === tempId 
                    ? { ...msg, content: data.content || '' } 
                    : msg
                ));
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e);
            }
          }
        }
      }
      
      // If we didn't get a session ID yet, get one from a regular request
      if (!sessionId) {
        const sessionResponse = await axios.post(`${API_URL}/chat`, {
          message: "Hello",
        }, {
          headers: { 'X-API-Key': API_KEY }
        });
        
        if (sessionResponse.data.session_id) {
          setSessionId(sessionResponse.data.session_id);
        }
      }
      
    } catch (err) {
      console.error('Error sending message:', err);
      setError(err.message || 'An error occurred while sending your message');
      
      // Remove the streaming message if it exists
      setMessages(prev => prev.filter(msg => !msg.isStreaming));
      
      // Add error message
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'I\'m sorry, I encountered an error. Please try again.', 
        sources: [],
        isError: true
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const resetConversation = async () => {
    try {
      await axios.post(`${API_URL}/reset`, {
        session_id: sessionId
      }, {
        headers: { 'X-API-Key': API_KEY }
      });
      
      // Reset messages
      setMessages([{ 
        role: 'assistant', 
        content: 'Conversation reset. How can I help you today?',
        sources: []
      }]);
      
      setError(null);
    } catch (err) {
      console.error('Error resetting conversation:', err);
      setError('Failed to reset conversation');
    }
  };

  const toggleChat = () => {
    setIsChatOpen(prev => !prev);
    setIsMinimized(false);
  };

  const toggleMinimize = () => {
    setIsMinimized(prev => !prev);
  };

  return (
    <div className="App">
      {/* Chat Widget Button */}
      {!isChatOpen && (
        <button className="chat-widget-button" onClick={toggleChat}>
          <FaComments />
          <span>Chat with Neutrino AI</span>
        </button>
      )}

      {/* Chat Widget */}
      {isChatOpen && (
        <div className={`chat-widget ${isMinimized ? 'minimized' : ''}`}>
          {/* Chat Header */}
          <div className="chat-widget-header">
            <div className="logo-container">
              <FaRobot className="logo-icon" />
              <h1>Neutrino AI</h1>
            </div>
            <div className="header-actions">
              <button
                className="action-button"
                onClick={toggleMinimize}
                title={isMinimized ? "Expand" : "Minimize"}
              >
                {isMinimized ? <FaPlus /> : <FaMinus />}
              </button>
              <button
                className="action-button close-button"
                onClick={toggleChat}
                title="Close Chat"
              >
                <FaTimes />
              </button>
            </div>
          </div>

          {/* Chat Body */}
          {!isMinimized && (
            <div className="chat-widget-body">
              <div className="messages-container">
                {messages.map((message, index) => (
                  <div key={index} className={`message-wrapper ${message.role}`}>
                    <ChatMessage message={message} />
                    {/* Removed source citation rendering */}
                  </div>
                ))}
                {isLoading && !messages[messages.length - 1]?.isStreaming && (
                  <div className="message-wrapper assistant">
                    <div className="chat-message">
                      <div className="typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>
                    </div>
                  </div>
                )}
                {error && (
                  <div className="error-message">
                    {error}
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
              
              <form className="input-form" onSubmit={handleSubmit}>
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask a question..."
                  disabled={isLoading}
                  ref={inputRef}
                />
                <button 
                  type="submit" 
                  disabled={isLoading || !input.trim()}
                  className={isLoading ? 'loading' : ''}
                >
                  <FaPaperPlane />
                </button>
              </form>
            </div>
          )}

          {/* Chat Footer */}
          <div className="chat-widget-footer">
            <div className="footer-buttons">
              <button
                className="reset-button"
                onClick={resetConversation}
                title="Reset conversation"
              >
                New Conversation
              </button>
              <button
                className="close-chat-button"
                onClick={toggleChat}
                title="Close the chat window"
              >
                Close Chat
              </button>
            </div>
            <div className="powered-by">
              Powered by Neutrino Tech
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;