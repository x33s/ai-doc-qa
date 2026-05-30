<template>
  <div class="upload-area" 
       :class="{ 'drag-over': isDragOver }"
       @dragover.prevent="isDragOver = true"
       @dragleave.prevent="isDragOver = false"
       @drop.prevent="handleDrop">
    
    <input type="file" ref="fileInput" @change="handleFileSelect" accept=".pdf,.txt,.docx,.doc" style="display:none">
    
    <div class="upload-content">
      <div class="icon">📁</div>
      <p>拖拽文件到这里，或 <button class="link-btn" @click="$refs.fileInput.click()">点击上传</button></p>
      <p class="hint">支持 PDF、TXT、Word 格式</p>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'

const emit = defineEmits(['upload-success'])
const isDragOver = ref(false)
const fileInput = ref(null)

const uploadFile = async (file) => {
  const formData = new FormData()
  formData.append('file', file)
  
  try {
    await axios.post('http://localhost:8000/upload', formData)
    emit('upload-success')
    alert(`文件 ${file.name} 上传成功！`)
  } catch (error) {
    console.error('上传失败', error)
    alert('上传失败，请重试')
  }
}

const handleFileSelect = (e) => {
  const file = e.target.files[0]
  if (file) uploadFile(file)
}

const handleDrop = (e) => {
  isDragOver.value = false
  const file = e.dataTransfer.files[0]
  if (file) uploadFile(file)
}
</script>

<style scoped>
.upload-area {
  border: 2px dashed #ccc;
  border-radius: 12px;
  padding: 40px 20px;
  text-align: center;
  margin: 16px;
  transition: all 0.3s;
  background: white;
}

.upload-area.drag-over {
  border-color: #667eea;
  background: #f0f0ff;
}

.icon {
  font-size: 48px;
  margin-bottom: 12px;
}

.link-btn {
  background: none;
  border: none;
  color: #667eea;
  cursor: pointer;
  font-size: 14px;
  text-decoration: underline;
}

.link-btn:hover {
  color: #764ba2;
}

.hint {
  font-size: 12px;
  color: #999;
  margin-top: 8px;
}
</style>