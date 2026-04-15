import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { HttpClient } from '@angular/common/http';

interface ChatMessage {
  text: string;
  sender: 'user' | 'bot';
}

@Component({
  selector: 'app-chatbot',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatCardModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatIconModule,
    MatProgressSpinnerModule
  ],
  templateUrl: './chatbot.html',
  styleUrl: './chatbot.css'
})
export class Chatbot implements OnInit {
  messages: ChatMessage[] = [];
  newMessage: string = '';
  isLoading: boolean = false;
  userId: string = '';

  constructor(private http: HttpClient) {}

  ngOnInit() {
    let storedId = localStorage.getItem('chat_user_id');
    if (!storedId) {
      storedId = typeof crypto !== 'undefined' && crypto.randomUUID 
        ? crypto.randomUUID() 
        : 'user-' + Math.random().toString(36).substr(2, 9);
      localStorage.setItem('chat_user_id', storedId);
    }
    this.userId = storedId;
    
    // Trigger the initial flow in the backend
    this.sendMessage('Hola', true);
  }

  sendMessage(overrideMessage?: string, isInitial: boolean = false) {
    const text = overrideMessage || this.newMessage.trim();
    if (!text) return;

    if (!isInitial) {
        this.messages.push({ text: text, sender: 'user' });
        this.newMessage = '';
    }
    this.isLoading = true;

    this.http.post<{response: string}>('/api/web-chat', {
      user_id: this.userId,
      message: text
    }).subscribe({
      next: (res) => {
        this.messages.push({ text: res.response, sender: 'bot' });
        this.isLoading = false;
      },
      error: (err) => {
        console.error('Chat error:', err);
        this.messages.push({ text: 'Error de conexión. Intenta de nuevo.', sender: 'bot' });
        this.isLoading = false;
      }
    });
  }
}