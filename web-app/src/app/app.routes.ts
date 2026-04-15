import { Routes } from '@angular/router';
import { AdminDashboard } from './admin-dashboard/admin-dashboard';
import { Chatbot } from './chatbot/chatbot';

export const routes: Routes = [
  { path: '', redirectTo: 'chat', pathMatch: 'full' },
  { path: 'admin', component: AdminDashboard },
  { path: 'chat', component: Chatbot }
];