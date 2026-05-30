<template>
  <div class="chat-window">
    <div class="messages" ref="messagesContainer">
      <div v-for="(msg, idx) in messages" :key="idx" class="message" :class="msg.role">
        <div class="avatar">
          {{ msg.role === 'user' ? '👤' : '🤖' }}
        </div>
        <div class="content">
          <div class="text">{{ msg.content }}</div>
          <div v-if="msg.sources && msg.sources.length" class="sources">
            <details>
              <summary>📚 引用来源</summary>
              <ul>
                <li v-for="(src, i) in msg.sources" :key="i">{{ src }}</li>
              </ul>
            </details>
          </div>
          <div class="time">{{ formatTime(msg.timestamp) }}</div>
        </div>
      </div>
      
      <div v-if="isThinking" class="message assistant thinking">
        <div class="avatar">🤖</div>
        <div class="content">
          <div class="dot-floating">🤔 思考中...</div>
        </div>
      </div>
    </div>
    
    <div class="input-area">
      <input 
        v-model="inputText" 
        @keydown.enter="send"
        placeholder="输入问题，按 Enter 发送..."
        :disabled="isThinking"
      />
      <button @click="send" :disabled="isThinking || !inputText.trim()">发送</button>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick, watch } from 'vue'

const props = defineProps({
  messages: Array,
  isThinking: Boolean
})

const emit = defineEmits(['send'])

const inputText = ref('')
const messagesContainer = ref(null)

const send = () => {
  if (!inputText.value.trim() || props.isThinking) return
  emit('send', inputText.value)
  inputText.value = ''
}

const formatTime = (timestamp) => {
  if (!timestamp) return ''
  const date = new Date(timestamp)
  return `${date.getHours().toString().padStart(2,'0')}:${date.getMinutes().toString().padStart(2,'0')}`
}

// 自动滚动到底部
watch(() => props.messages.length, async () => {
  await nextTick()
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
})
</script>

<style scoped>
.chat-window {
  flex: 1;
  display: flex;
  flex-direction: column;
  height: 100%;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}

.message {
  display: flex;
  margin-bottom: 20px;
  animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.message.user {
  flex-direction: row-reverse;
}

.message.user .avatar {
  background: #667eea;
}

.message.assistant .avatar {
  background: #764ba2;
}

.avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #e0e0e0;
  margin: 0 12px;
  flex-shrink: 0;
}

.content {
  max-width: 70%;
  background: white;
  padding: 12px 16px;
  border-radius: 12px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

.message.user .content {
  background: #667eea;
  color: white;
}

.text {
  white-space: pre-wrap;
  word-break: break-word;
}

.sources {
  margin-top: 8px;
  font-size: 12px;
  opacity: 0.7;
}

.sources details summary {
  cursor: pointer;
  outline: none;
}

.sources ul {
  margin: 8px 0 0 20px;
  padding: 0;
}

.time {
  font-size: 10px;
  opacity: 0.5;
  margin-top: 4px;
}

.thinking .content {
  background: #e0e0e0;
  color: #666;
}

.dot-floating {
  animation: pulse 1.5s ease infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 0.5; }
  50% { opacity: 1; }
}

.input-area {
  display: flex;
  padding: 16px;
  background: white;
  border-top: 1px solid #e0e0e0;
  gap: 12px;
}

.input-area input {
  flex: 1;
  padding: 12px;
  border: 1px solid #e0e0e0;
  border-radius: 24px;
  outline: none;
  font-size: 14px;
}

.input-area input:focus {
  border-color: #667eea;
}

.input-area button {
  padding: 8px 24px;
  background: #667eea;
  color: white;
  border: none;
  border-radius: 24px;
  cursor: pointer;
  font-size: 14px;
}

.input-area button:hover:not(:disabled) {
  background: #764ba2;
}

.input-area button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>