import RAGChat from './components/RAGChat';

function App() {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <nav className="bg-white shadow-sm p-4">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <h1 className="text-xl font-bold">Kushagra's Professional Profile Assistant</h1>
          <a href="https://github.com/KushagraSikka" 
             target="_blank" 
             rel="noopener noreferrer"
             className="text-gray-600 hover:text-gray-900">
            GitHub
          </a>
        </div>
      </nav>
      <main className="flex-1 container mx-auto p-4">
        <RAGChat />
      </main>
    </div>
  );
}

export default App;