import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { MatTableModule } from '@angular/material/table';
import { MatPaginatorModule, PageEvent } from '@angular/material/paginator';
import { MatCardModule } from '@angular/material/card';
import { MatChipsModule } from '@angular/material/chips';
import { CommonModule } from '@angular/common';

interface Profile {
  user_id: string;
  status: string;
  level: string;
  step: number;
  messages: number;
}

interface ApiResponse {
  profiles: Profile[];
  total_pages: number;
  current_page: number;
  total_records: number;
}

@Component({
  selector: 'app-admin-dashboard',
  standalone: true,
  imports: [CommonModule, MatTableModule, MatPaginatorModule, MatCardModule, MatChipsModule],
  templateUrl: './admin-dashboard.html',
  styleUrl: './admin-dashboard.css'
})
export class AdminDashboard implements OnInit {
  displayedColumns: string[] = ['user_id', 'status', 'level', 'step', 'messages'];
  dataSource: Profile[] = [];
  totalRecords = 0;
  pageSize = 10;
  pageIndex = 0;

  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.loadData();
  }

  loadData() {
    this.http.get<ApiResponse>(`/api/profiles?page=${this.pageIndex + 1}&size=${this.pageSize}`, {
      withCredentials: true
    }).subscribe({
      next: (data) => {
        this.dataSource = data.profiles;
        this.totalRecords = data.total_records;
      },
      error: (err) => {
        console.error('Error fetching profiles', err);
        // Prompt for manual basic auth check if needed
        if (err.status === 401) {
            alert('Autenticación requerida para acceder al dashboard.');
        }
      }
    });
  }

  onPageChange(event: PageEvent) {
    this.pageIndex = event.pageIndex;
    this.pageSize = event.pageSize;
    this.loadData();
  }

  getStatusColor(status: string): string {
    return status === 'applying' ? 'accent' : 'primary';
  }
}