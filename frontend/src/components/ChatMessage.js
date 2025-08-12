import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import remarkGfm from 'remark-gfm';
import { FaUser, FaRobot } from 'react-icons/fa';

const ChatMessage = ({ message }) => {
  const { role, content, isStreaming } = message;
  
  // Function to simplify markdown content if needed
  const simplifyContent = (content) => {
    // For now, just return the content as is
    // In the future, you could add logic to simplify complex markdown
    return content;
  };
  
  // Custom renderer for code blocks
  const components = {
    code({ node, inline, className, children, ...props }) {
      const match = /language-(\w+)/.exec(className || '');
      return !inline && match ? (
        <SyntaxHighlighter
          style={atomDark}
          language={match[1]}
          PreTag="div"
          {...props}
        >
          {String(children).replace(/\n$/, '')}
        </SyntaxHighlighter>
      ) : (
        <code className={className} {...props}>
          {children}
        </code>
      );
    }
  };

  return (
    <div className="chat-message">
      <div className="message-header">
        {role === 'user' ? (
          <FaUser className="message-icon user-icon" />
        ) : (
          <FaRobot className="message-icon assistant-icon" />
        )}
        <span className="message-role">
          {role === 'user' ? 'You' : 'Neutrino AI'}
        </span>
      </div>
      
      <div className="message-content">
        {content ? (
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={components}
          >
            {simplifyContent(content)}
          </ReactMarkdown>
        ) : isStreaming ? (
          <div className="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        ) : null}
      </div>
    </div>
  );
};

export default ChatMessage;