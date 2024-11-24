import RAGDashboard from './components/RAGDashboard';
import { Github } from 'lucide-react';

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
          <h1 className="text-xl font-bold">Professional Profile Assistant</h1>
          <a
            href="https://github.com/KushagraSikka"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-gray-600 hover:text-gray-900"
          >
            <Github className="w-5 h-5" />
            <span>GitHub</span>
          </a>
        </div>
      </nav>
      <main className="py-6">
        <RAGDashboard />
      </main>
    </div>
  );
}

export default App;