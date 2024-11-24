import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Search, MessageSquare, Workflow, Terminal, Database, ChevronRight } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface ProcessingMetrics {
  timestamp: number;
  tokens_processed: number;
  step: string;
}

interface TerminalLog {
  timestamp: number;
  message: string;
  type: 'info' | 'error' | 'success';
}

export const RAGDashboard = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeStep, setActiveStep] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<ProcessingMetrics[]>([]);
  const [logs, setLogs] = useState<TerminalLog[]>([]);

  const steps = [
    { id: 'query', icon: MessageSquare, label: 'Query Processing', color: 'blue' },
    { id: 'embedding', icon: Brain, label: 'Text Embedding', color: 'purple' },
    { id: 'retrieval', icon: Search, label: 'Document Retrieval', color: 'green' },
    { id: 'response', icon: Workflow, label: 'Response Generation', color: 'orange' }
  ];

  const addLog = (message: string, type: 'info' | 'error' | 'success' = 'info') => {
    setLogs(prev => [...prev, { timestamp: Date.now(), message, type }]);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    setLoading(true);
    setMessages(prev => [...prev, { role: 'user', content: input }]);
    const startTime = Date.now();

    try {
      for (const step of steps) {
        setActiveStep(step.id);
        addLog(`Processing ${step.label}...`, 'info');
        await new Promise(r => setTimeout(r, 800));
      }

      const response = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: input })
      });

      const data = await response.json();
      setMessages(prev => [...prev, { role: 'assistant', content: data.answer }]);
      setMetrics(data.metrics || []);
      addLog('Response generated successfully', 'success');
    } catch (error) {
      addLog(`Error: ${error}`, 'error');
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.'
      }]);
    } finally {
      setLoading(false);
      setInput('');
      setActiveStep(null);
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-6">
          {/* Pipeline visualization */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
              <Database className="w-6 h-6" />
              RAG Pipeline
            </h2>
            <div className="space-y-4">
              {steps.map((step, i) => (
                <motion.div
                  key={step.id}
                  className={`flex items-center p-4 rounded-lg transition-all duration-300 ${
                    activeStep === step.id ? `bg-${step.color}-50 shadow-lg` : 'bg-gray-50'
                  }`}
                  animate={{
                    scale: activeStep === step.id ? 1.02 : 1,
                    opacity: activeStep === step.id ? 1 : 0.7
                  }}
                >
                  <step.icon className={`w-6 h-6 ${
                    activeStep === step.id ? `text-${step.color}-500` : 'text-gray-500'
                  }`} />
                  <span className="ml-3 font-medium">{step.label}</span>
                  {activeStep === step.id && (
                    <motion.div 
                      className={`ml-auto w-2 h-2 rounded-full bg-${step.color}-500`}
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ repeat: Infinity, duration: 1 }}
                    />
                  )}
                </motion.div>
              ))}
            </div>
          </div>

          {/* Terminal */}
          <div className="bg-gray-900 rounded-lg shadow-lg p-4 font-mono text-sm">
            <div className="flex items-center gap-2 mb-4 text-gray-400">
              <Terminal className="w-4 h-4" />
              <span>System Logs</span>
            </div>
            <div className="h-48 overflow-y-auto space-y-2">
              <AnimatePresence>
                {logs.map((log, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    className={`flex items-start gap-2 ${
                      log.type === 'error' ? 'text-red-400' :
                      log.type === 'success' ? 'text-green-400' :
                      'text-gray-300'
                    }`}
                  >
                    <ChevronRight className="w-4 h-4 mt-1 flex-shrink-0" />
                    <span>{log.message}</span>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </div>
        </div>

        <div className="space-y-6">
          {/* Chat interface */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-bold mb-6">Ask About Kushagra</h2>
            <div className="h-96 overflow-y-auto mb-4 space-y-4">
              <AnimatePresence>
                {messages.map((msg, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className={`flex ${
                      msg.role === 'user' ? 'justify-end' : 'justify-start'
                    }`}
                  >
                    <div className={`max-w-[80%] p-3 rounded-lg ${
                      msg.role === 'user' 
                        ? 'bg-blue-500 text-white' 
                        : 'bg-gray-100'
                    }`}>
                      {msg.content}
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
            
            <form onSubmit={handleSubmit} className="flex gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about skills, experience, or projects..."
                className="flex-1 p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={loading}
              />
              <button
                type="submit"
                disabled={loading}
                className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-400 transition-colors"
              >
                {loading ? 'Processing...' : 'Send'}
              </button>
            </form>
          </div>

          {/* Metrics */}
          {metrics.length > 0 && (
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-2xl font-bold mb-6">Processing Metrics</h2>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={metrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="timestamp" 
                      tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                    />
                    <YAxis />
                    <Tooltip 
                      labelFormatter={(value) => new Date(value).toLocaleTimeString()}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="tokens_processed" 
                      stroke="#8884d8" 
                      name="Tokens"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RAGDashboard;