import axios from "axios";

const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

export interface Question {
  question: string;
}

export interface Answer {
  answer: string;
}

export const ragService = {
  async askQuestion(question: string): Promise<Answer> {
    const response = await api.post<Answer>("/ask", { question });
    return response.data;
  },

  async checkHealth(): Promise<{ status: string }> {
    const response = await api.get<{ status: string }>("/health");
    return response.data;
  },
};

export default ragService;
