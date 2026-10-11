"""Session-based Conversational Memory Manager for Universal RAG.

Maintains isolated conversation histories and metadata per session_id using LangChain message objects.
Keeps chat memory strictly separate from the RAG document vector store.
"""

import threading
import time
from typing import Any, Dict, List, Optional
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class SessionMemoryManager:
    """
    Thread-safe in-memory session manager with sliding-window history and session metadata.
    Stores HumanMessage and AIMessage pairs isolated by session_id.
    """

    def __init__(self, max_messages: int = 30):
        self._sessions: Dict[str, List[BaseMessage]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.max_messages = max_messages

    def get_history(self, session_id: str) -> List[BaseMessage]:
        """Retrieve the recent message history for a given session."""
        with self._lock:
            if session_id not in self._sessions:
                return []
            return list(self._sessions[session_id])

    def get_session_messages(self, session_id: str) -> List[Dict[str, str]]:
        """Retrieve message history as a clean JSON-serializable list of message dicts."""
        history = self.get_history(session_id)
        result = []
        for msg in history:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            result.append({"role": role, "content": msg.content})
        return result

    def add_turn(self, session_id: str, human_text: str, ai_text: str) -> None:
        """Append a human question and assistant answer turn to the session history."""
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = []

            now = time.time()
            if session_id not in self._metadata:
                clean_title = human_text.strip().replace("\n", " ")
                if len(clean_title) > 36:
                    clean_title = clean_title[:33] + "..."
                self._metadata[session_id] = {
                    "session_id": session_id,
                    "title": clean_title or "New Conversation",
                    "created_at": now,
                    "updated_at": now,
                    "message_count": 0,
                }

            history = self._sessions[session_id]
            history.append(HumanMessage(content=human_text))
            history.append(AIMessage(content=ai_text))

            # Maintain sliding window (keep only the most recent N messages)
            if len(history) > self.max_messages:
                self._sessions[session_id] = history[-self.max_messages :]

            self._metadata[session_id]["updated_at"] = now
            self._metadata[session_id]["message_count"] = len(self._sessions[session_id])

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Return all active conversation sessions sorted by most recent activity."""
        with self._lock:
            # Ensure any active sessions have at least minimal metadata
            sessions_list = []
            for sid, meta in self._metadata.items():
                sessions_list.append({
                    "session_id": sid,
                    "title": meta.get("title", "Conversation"),
                    "created_at": meta.get("created_at", 0),
                    "updated_at": meta.get("updated_at", 0),
                    "message_count": meta.get("message_count", 0),
                })
            # Also catch any session in _sessions not yet in metadata
            for sid, msgs in self._sessions.items():
                if sid not in self._metadata and msgs:
                    sessions_list.append({
                        "session_id": sid,
                        "title": "Chat " + sid[:6],
                        "created_at": 0,
                        "updated_at": 0,
                        "message_count": len(msgs),
                    })

            sessions_list.sort(key=lambda s: s["updated_at"], reverse=True)
            return sessions_list

    def clear_session(self, session_id: str) -> bool:
        """Clear the message history for a specific session."""
        with self._lock:
            found = False
            if session_id in self._sessions:
                self._sessions[session_id] = []
                found = True
            if session_id in self._metadata:
                self._metadata[session_id]["message_count"] = 0
                self._metadata[session_id]["updated_at"] = time.time()
                found = True
            return found

    def delete_session(self, session_id: str) -> bool:
        """Completely remove a session and its metadata."""
        with self._lock:
            found = False
            if session_id in self._sessions:
                del self._sessions[session_id]
                found = True
            if session_id in self._metadata:
                del self._metadata[session_id]
                found = True
            return found

    def clear_all(self) -> None:
        """Clear all session histories and metadata."""
        with self._lock:
            self._sessions.clear()
            self._metadata.clear()

    def get_formatted_history(self, session_id: str) -> str:
        """Format the conversation history as a readable string for query rewriting."""
        messages = self.get_history(session_id)
        if not messages:
            return ""

        formatted_lines = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_lines.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted_lines.append(f"Assistant: {msg.content}")
        return "\n".join(formatted_lines)


# Singleton memory manager instance
session_memory = SessionMemoryManager(max_messages=30)
