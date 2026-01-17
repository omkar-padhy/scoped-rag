"""
Chat Management System
Handles chat persistence, history, and context management for all users.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import uuid

# Storage path for chat data
CHAT_STORAGE_PATH = Path("chat_data")


def ensure_storage_exists():
    """Create chat storage directory if it doesn't exist"""
    CHAT_STORAGE_PATH.mkdir(exist_ok=True)


def get_user_chats_path(username: str) -> Path:
    """Get the path to a user's chat storage file"""
    ensure_storage_exists()
    return CHAT_STORAGE_PATH / f"{username}_chats.json"


def load_user_chats(username: str) -> dict:
    """Load all chats for a user"""
    path = get_user_chats_path(username)
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"chats": {}, "active_chat_id": None}
    return {"chats": {}, "active_chat_id": None}


def save_user_chats(username: str, data: dict):
    """Save all chats for a user"""
    path = get_user_chats_path(username)
    ensure_storage_exists()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def create_new_chat(username: str, title: Optional[str] = None) -> str:
    """Create a new chat session and return its ID"""
    data = load_user_chats(username)
    
    chat_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().isoformat()
    
    data["chats"][chat_id] = {
        "id": chat_id,
        "title": title or f"Chat {len(data['chats']) + 1}",
        "created_at": timestamp,
        "updated_at": timestamp,
        "messages": [],
        "context": {
            "model_used": None,
            "total_queries": 0,
            "sources_used": []
        }
    }
    data["active_chat_id"] = chat_id
    
    save_user_chats(username, data)
    return chat_id


def get_chat(username: str, chat_id: str) -> Optional[dict]:
    """Get a specific chat by ID"""
    data = load_user_chats(username)
    return data["chats"].get(chat_id)


def get_active_chat(username: str) -> Optional[dict]:
    """Get the active chat for a user"""
    data = load_user_chats(username)
    active_id = data.get("active_chat_id")
    if active_id and active_id in data["chats"]:
        return data["chats"][active_id]
    return None


def set_active_chat(username: str, chat_id: str) -> bool:
    """Set the active chat for a user"""
    data = load_user_chats(username)
    if chat_id in data["chats"]:
        data["active_chat_id"] = chat_id
        save_user_chats(username, data)
        return True
    return False


def add_message_to_chat(
    username: str, 
    chat_id: str, 
    role: str, 
    content: str,
    sources: Optional[list] = None,
    context: Optional[str] = None,
    model: Optional[str] = None
) -> bool:
    """Add a message to a chat"""
    data = load_user_chats(username)
    
    if chat_id not in data["chats"]:
        return False
    
    chat = data["chats"][chat_id]
    
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(),
    }
    
    if sources:
        message["sources"] = sources
    if context:
        message["context"] = context
    if model:
        message["model"] = model
        chat["context"]["model_used"] = model
    
    chat["messages"].append(message)
    chat["updated_at"] = datetime.now().isoformat()
    
    # Update chat context
    if role == "assistant":
        chat["context"]["total_queries"] += 1
        if sources:
            for src in sources:
                if src not in chat["context"]["sources_used"]:
                    chat["context"]["sources_used"].append(src)
    
    # Auto-generate title from first user message
    if role == "user" and len(chat["messages"]) == 1:
        # Use first 50 chars of first message as title
        chat["title"] = content[:50] + ("..." if len(content) > 50 else "")
    
    save_user_chats(username, data)
    return True


def get_chat_messages(username: str, chat_id: str) -> list:
    """Get all messages from a chat"""
    chat = get_chat(username, chat_id)
    if chat:
        return chat.get("messages", [])
    return []


def list_user_chats(username: str) -> list:
    """List all chats for a user (sorted by updated_at desc)"""
    data = load_user_chats(username)
    chats = list(data["chats"].values())
    # Sort by updated_at descending (most recent first)
    chats.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return chats


def delete_chat(username: str, chat_id: str) -> bool:
    """Delete a chat"""
    data = load_user_chats(username)
    
    if chat_id not in data["chats"]:
        return False
    
    del data["chats"][chat_id]
    
    # If we deleted the active chat, set active to most recent or None
    if data["active_chat_id"] == chat_id:
        if data["chats"]:
            # Set to most recent chat
            sorted_chats = sorted(
                data["chats"].values(), 
                key=lambda x: x.get("updated_at", ""), 
                reverse=True
            )
            data["active_chat_id"] = sorted_chats[0]["id"]
        else:
            data["active_chat_id"] = None
    
    save_user_chats(username, data)
    return True


def clear_all_chats(username: str):
    """Clear all chats for a user"""
    data = {"chats": {}, "active_chat_id": None}
    save_user_chats(username, data)


def get_chat_context_summary(username: str, chat_id: str) -> dict:
    """Get a summary of the chat context"""
    chat = get_chat(username, chat_id)
    if not chat:
        return {}
    
    return {
        "title": chat.get("title"),
        "created_at": chat.get("created_at"),
        "updated_at": chat.get("updated_at"),
        "message_count": len(chat.get("messages", [])),
        "model_used": chat.get("context", {}).get("model_used"),
        "total_queries": chat.get("context", {}).get("total_queries", 0),
        "unique_sources": len(chat.get("context", {}).get("sources_used", []))
    }


def export_chat(username: str, chat_id: str) -> Optional[str]:
    """Export a chat as formatted text"""
    chat = get_chat(username, chat_id)
    if not chat:
        return None
    
    lines = [
        f"Chat: {chat['title']}",
        f"Created: {chat['created_at']}",
        f"Model: {chat.get('context', {}).get('model_used', 'N/A')}",
        "-" * 50,
        ""
    ]
    
    for msg in chat.get("messages", []):
        role = "You" if msg["role"] == "user" else "Assistant"
        lines.append(f"[{role}] ({msg.get('timestamp', 'N/A')[:19]})")
        lines.append(msg["content"])
        if msg.get("sources"):
            lines.append(f"Sources: {', '.join(msg['sources'][:3])}")
        lines.append("")
    
    return "\n".join(lines)
