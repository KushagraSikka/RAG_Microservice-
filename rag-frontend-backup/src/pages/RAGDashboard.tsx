import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  MessageSquare, 
  Database, 
  Brain, 
  Search, 
  Workflow, 
  ArrowRight, 
  Send,
  AlertCircle,
  CheckCircle2,
  Timer,
  Hash
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: number;
}

interface ProcessingMetric {
  timestamp: number;
  tokens_processed: number;
  processing_time: number;
  step: string;
}

const RAGDashboard = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeStep, setActiveStep] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<'connected' | 'disconnected'>('disconnected');
  const [metrics, setMetrics] = useState<ProcessingMetric[]>([]);
  const [processingStats, setProcessingStats] = useState({
    totalTokens: 0,
    avgProcessingTime: 0,
    successRate: 100,
  });

  const pipelineSteps = [
    { 
      id: 'input', 
      icon: MessageSquare, 
      label: 'User Input', 
      description: 'Query preprocessing',
      color: 'blue'
    },
    { 
      id: 'embedding', 
      icon: Brain, 
      label: 'Text Embedding', 
      description: 'Vector transformation',
      color: 'purple'
    },
    { 
      id: 'retrieval', 
      icon: Search, 
      label: 'Document Retrieval', 
      description: 'Context matching',
      color: 'green'
    },
    { 
      id: 'generation', 
      icon: Workflow, 
      label: 'Response Generation', 
      description: 'Answer creation',
      color: 'orange'
    }
  ];

  useEffect(() => {
    checkBackendStatus();
    const interval = setInterval(checkBackendStatus, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const checkBackendStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/health');
      setBackendStatus(response.ok ? 'connected' : 'disconnected');
    } catch (error) {
      setBackendStatus('disconnected');
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    setLoading(true);
    const startTime = Date.now();
    const newMessage: Message = { 
      role: 'user', 
      content: input,
      timestamp: startTime 
    };
    setMessages(prev => [...prev, newMessage]);

    try {
      // Simulate pipeline steps
      for (const step of pipelineSteps) {
        setActiveStep(step.id);
        await new Promise(resolve => setTimeout(resolve, 800));
      }

      const response = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({ question: input })
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();
      
      // Update metrics
      if (data.metrics) {
        setMetrics(data.metrics);
        updateProcessingStats(data.metrics);
      }

      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: data.answer,
        timestamp: Date.now()
      }]);

    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error processing your request.',
        timestamp: Date.now()
      }]);
    } finally {
      setLoading(false);
      setInput('');
      setActiveStep(null);
    }
  };

  const updateProcessingStats = (newMetrics: ProcessingMetric[]) => {
    const lastMetric = newMetrics[newMetrics.length - 1];
    setProcessingStats(prev => ({
      totalTokens: prev.totalTokens + lastMetric.tokens_processed,
      avgProcessingTime: (prev.avgProcessingTime + lastMetric.processing_time) / 2,
      successRate: messages.length > 0 ? 
        (messages.filter(m => !m.content.includes('error')).length / messages.length) * 100 : 
        100
    }));
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4">
        {/* Status Bar */}
        <div className="mb-6 flex items-center justify-between bg-white rounded-lg p-4 shadow-sm">
          <div className="flex items-center space-x-4">
            <div className={`flex items-center gap-2 ${
              backendStatus === 'connected' ? 'text-green-500' : 'text-red-500'
            }`}>
              {backendStatus === 'connected' ? 
                <CheckCircle2 className="w-5 h-5" /> : 
                <AlertCircle className="w-5 h-5" />}
              <span className="font-medium">
                Backend {backendStatus === 'connected' ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            <div className="flex items-center gap-2 text-blue-500">
              <Timer className="w-5 h-5" />
              <span className="font-medium">
                Avg. Response: {processingStats.avgProcessingTime.toFixed(2)}s
              </span>
            </div>
            <div className="flex items-center gap-2 text-purple-500">
              <Hash className="w-5 h-5" />
              <span className="font-medium">
                Total Tokens: {processingStats.totalTokens}
              </span>
            </div>
          </div>
          <div className="text-gray-500 text-sm">
            Success Rate: {processingStats.successRate.toFixed(1)}%
          </div>
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Pipeline Visualization */}
          <div className="space-y-8">
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-2xl font-semibold mb-6">Pipeline Architecture</h2>
              
              <div className="space-y-4">
                {pipelineSteps.map((step, index) => (
                  <motion.div
                    key={step.id}
                    className={`flex items-center p-4 rounded-lg transition-all duration-300 
                      ${activeStep === step.id ? `bg-${step.color}-50 shadow-md` : 'bg-gray-50'}`}
                    animate={{
                      scale: activeStep === step.id ? 1.02 : 1,
                      opacity: activeStep === step.id ? 1 : 0.7
                    }}
                  >
                    <step.icon 
                      className={`w-8 h-8 ${
                        activeStep === step.id ? `text-${step.color}-500` : 'text-gray-500'
                      }`} 
                    />
                    {index < pipelineSteps.length - 1 && (
                      <ArrowRight className={`w-6 h-6 mx-2 text-gray-400 transition-opacity duration-300 ${
                        activeStep === step.id ? 'opacity-100' : 'opacity-50'
                      }`} />
                    )}
                    <div className="ml-4">
                      <h3 className="font-medium">{step.label}</h3>
                      <p className="text-sm text-gray-600">{step.description}</p>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Processing Metrics */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-2xl font-semibold mb-4">Processing Metrics</h2>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={metrics}>
                    <defs>
                      <linearGradient id="tokenGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#8884d8" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#8884d8" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="timestamp" 
                      tickFormatter={(value) => `${(value / 1000).toFixed(1)}s`}
                    />
                    <YAxis yAxisId="tokens" />
                    <YAxis yAxisId="time" orientation="right" />
                    <Tooltip 
                      labelFormatter={(value) => `Time: ${(value / 1000).toFixed(1)}s`}
                      formatter={(value: number, name: string) => [
                        name === 'Processing Time' ? `${value.toFixed(2)}s` : value,
                        name
                      ]}
                    />
                    <Area
                      yAxisId="tokens"
                      type="monotone"
                      dataKey="tokens_processed"
                      stroke="#8884d8"
                      fillOpacity={1}
                      fill="url(#tokenGradient)"
                      name="Tokens Processed"
                    />
                    <Line 
                      yAxisId="time"
                      type="monotone"
                      dataKey="processing_time"
                      stroke="#82ca9d"
                      name="Processing Time"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Chat Interface */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold mb-4">Interactive Chat</h2>
            
            <div className="h-[600px] overflow-y-auto mb-4 space-y-4 p-4">
              <AnimatePresence>
                {messages.map((message, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className={`p-4 rounded-lg ${
                      message.role === 'user' 
                        ? 'bg-blue-50 ml-12' 
                        : 'bg-gray-50 mr-12'
                    }`}
                  >
                    <div className="flex justify-between items-start">
                      <div className="flex-1">{message.content}</div>
                      {message.timestamp && (
                        <div className="text-xs text-gray-400 ml-2">
                          {new Date(message.timestamp).toLocaleTimeString()}
                        </div>
                      )}
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
                placeholder="Ask about Rick and Morty episodes..."
                className="flex-1 p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={loading}
              />
              <button
                type="submit"
                disabled={loading}
                className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 
                  disabled:bg-gray-400 transition-colors duration-200 flex items-center gap-2"
              >
                {loading ? (
                  <>
                    <div className="animate-spin h-5 w-5 border-2 border-white border-t-transparent rounded-full" />
                    <span>Processing...</span>
                  </>
                ) : (
                  <>
                    <Send className="w-5 h-5" />
                    <span>Send</span>
                  </>
                )}
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RAGDashboard;