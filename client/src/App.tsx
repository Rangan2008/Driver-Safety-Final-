/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';

// Components
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import Problem from './components/Problem';
import Solution from './components/Solution';
import Response from './components/Response';
import CTA from './components/CTA';
import Footer from './components/Footer';
import Profile from './components/Profile';
import ModelShowcase from './components/ModelShowcase';
import Docs from './components/Docs';
import Login from './components/Login';

export default function App() {
  const [currentView, setCurrentView] = useState<'home' | 'profile' | 'showcase' | 'docs' | 'login'>('home');
  const [isLightMode, setIsLightMode] = useState(false);

  const toggleLightMode = () => {
    setIsLightMode(!isLightMode);
    if (!isLightMode) {
      document.documentElement.classList.add('light');
    } else {
      document.documentElement.classList.remove('light');
    }
  };

  const handleViewChange = (view: 'home' | 'profile' | 'showcase' | 'docs' | 'login') => {
    setCurrentView(view);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen bg-background text-foreground selection:bg-primary/30">
      <Navbar currentView={currentView} setCurrentView={setCurrentView} isLightMode={isLightMode} toggleLightMode={toggleLightMode} />
      <main className="overflow-hidden">
        <AnimatePresence mode="wait">
          {currentView === 'home' && (
            <motion.div
              key="home"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <Hero setCurrentView={handleViewChange} />
              <Problem />
              <Solution />
              <Response />
              <CTA />
            </motion.div>
          )}
          {currentView === 'profile' && (
            <motion.div
              key="profile"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <Profile />
            </motion.div>
          )}
          {currentView === 'showcase' && (
            <motion.div
              key="showcase"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <ModelShowcase />
            </motion.div>
          )}
          {currentView === 'docs' && (
            <motion.div
              key="docs"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <Docs />
            </motion.div>
          )}
          {currentView === 'login' && (
            <motion.div
              key="login"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <Login setCurrentView={handleViewChange} />
            </motion.div>
          )}
        </AnimatePresence>
      </main>
      <Footer />
    </div>
  );
}
