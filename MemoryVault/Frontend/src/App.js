import React from "react";
import { BrowserRouter as Router, Route, Link, Routes } from "react-router-dom";
import "./App.css";
import QueryPage from "./components/QueryPage";
import MemoryInput from "./components/MemoryInput";

function App() {
  return (
    <Router>
      <div className="App">
        <header className="App-header">
          <nav className="nav-menu">
            <Link to="/" className="nav-link" id="queryMemories">
              Query Memories
            </Link>
            <div className="logo-title-container">
              <img
                src="/logo.png"
                alt="MemoryVault Logo"
                className="nav-logo"
              />
              <span className="nav-title" id="memoryVault">
                MemoryVault
              </span>
            </div>
            <Link to="/add-memory" className="nav-link" id="addMemory">
              Add Memory
            </Link>
          </nav>
        </header>

        <Routes>
          <Route path="/" element={<QueryPage />} />
          <Route path="/add-memory" element={<MemoryInput />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
