<template>
  <div class="app">
    <header>
      <h1>🤖 智能文档问答助手</h1>
      <p>上传文档，基于内容智能问答</p>
    </header>

    <div class="container">
      <!-- 左侧：文档管理 -->
      <div class="sidebar">
        <FileUpload @upload-success="onUploadSuccess" />
        
        <div class="doc-list">
          <h3>📄 已上传文档</h3>
          <ul>
            <li v-for="doc in documents" :key="doc">
              <span>{{ doc }}</span>
              <button @click="deleteDocument(doc)" class="delete-btn">删除</button>
            </li>
          </ul>
          <p v-if="documents.length ===0" class="empty">暂无文档</p>
        </div>
      </div>

      <!-- 右侧：对话窗口 -->
      <div class="chat-area">
        <ChatWindow 
          :messages="messages" 
          :is-thinking="isThinking"
          @send="sendMessage"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'
import FileUpload from './components/FileUpload.vue'
import ChatWindow from './components/ChatWindow.vue'

// API 基础地址
const API_BASE = 'http://localhost:8000'

// 数据
const documents = ref([])
const messages = ref([])
const isThinking = ref(false)

// 加载文档列表
const loadDocuments = async () => {
  try {
    const res = await axios.get(`${API_BASE}/documents`)
    documents.value = res.data.documents
  } catch (error) {
    console.error('加载文档失败', error)
  }
}

// 上传成功回调
const onUploadSuccess = () => {
  loadDocuments()
}

// 删除文档
const deleteDocument = async (filename) => {
  if (!confirm(`确定删除 ${filename} 吗？`)) return
  
  try {
    await axios.delete(`${API_BASE}/documents/${filename}`)
    loadDocuments()
    // 添加系统提示
    messages.value.push({
      role: 'system',
      content: `已删除文档：${filename}`,
      timestamp: new Date()
    })
  } catch (error) {
    console.error('删除失败', error)
  }
}

// 发送消息
const sendMessage = async (question) => {
  // 添加用户消息
  messages.value.push({
    role: 'user',
    content: question,
    timestamp: new Date()
  })
  
  isThinking.value = true
  
  try {
    const res = await axios.post(`${API_BASE}/chat`, {
      question: question,
      history: messages.value.filter(m => m.role !== 'system').slice(-5)
    })
    
    // 添加 AI 回答
    messages.value.push({
      role: 'assistant',
      content: res.data.answer,
      sources: res.data.sources,
      timestamp: new Date()
    })
  } catch (error) {
    console.error('问答失败', error)
    messages.value.push({
      role: 'assistant',
      content: '抱歉，处理问题时出错了。请确保已上传文档。',
      timestamp: new Date()
    })
  } finally {
    isThinking.value = false
  }
}

onMounted(() => {
  loadDocuments()
})
</script>

<style scoped>
.app {
  height: 100vh;
  display: flex;
  flex-direction: column;
}

header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 20px;
  text-align: center;
}

header h1 {
  margin: 0;
  font-size: 24px;
}

header p {
  margin: 8px 0 0;
  opacity: 0.9;
  font-size: 14px;
}

.container {
  flex: 1;
  display: flex;
  overflow: hidden;
}

.sidebar {
  width: 280px;
  background: #f5f5f5;
  border-right: 1px solid #e0e0e0;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
}

.chat-area {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: #fafafa;
}

.doc-list {
  padding: 16px;
  border-top: 1px solid #e0e0e0;
}

.doc-list h3 {
  margin: 0 0 12px 0;
  font-size: 14px;
  color: #666;
}

.doc-list ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

.doc-list li {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  background: white;
  margin-bottom: 8px;
  border-radius: 6px;
  font-size: 13px;
  word-break: break-all;
}

.doc-list li span {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.delete-btn {
  background: #ff4444;
  color: white;
  border: none;
  padding: 4px 8px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 11px;
  margin-left: 8px;
}

.delete-btn:hover {
  background: #cc0000;
}

.empty {
  color: #999;
  font-size: 13px;
  text-align: center;
  padding: 20px;
}
</style>