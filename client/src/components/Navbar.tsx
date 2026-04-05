import React from 'react';
import { Shield, Moon, Sun, User, LogIn } from 'lucide-react';

interface NavbarProps {
  currentView: string;
  setCurrentView: (v: 'home' | 'profile' | 'showcase' | 'docs' | 'login') => void;
  isLightMode: boolean;
  toggleLightMode: () => void;
}

const Navbar = ({ currentView, setCurrentView, isLightMode, toggleLightMode }: NavbarProps) => {
  const handleNavClick = (e: React.MouseEvent<HTMLAnchorElement>, view: 'home' | 'profile' | 'showcase' | 'docs' | 'login', hashId?: string) => {
    e.preventDefault();
    
    if (currentView !== view) {
      setCurrentView(view);
      if (hashId) {
        // Wait for the page transition animation to complete before scrolling
        setTimeout(() => {
          const el = document.getElementById(hashId);
          if (el) el.scrollIntoView({ behavior: 'smooth' });
        }, 350);
      } else {
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    } else {
      if (hashId) {
        const el = document.getElementById(hashId);
        if (el) el.scrollIntoView({ behavior: 'smooth' });
      } else {
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    }
  };

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-4 bg-background/80 backdrop-blur-md border-b border-white/10">
      <div 
        className="flex items-center gap-2 cursor-pointer"
        onClick={() => { setCurrentView('home'); window.scrollTo({ top: 0, behavior: 'smooth' }); }}
      >
        <Shield className="w-8 h-8 text-primary" />
        <span className="text-xl font-bold tracking-tight font-display">DriveSafe</span>
      </div>
      <div className="hidden md:flex items-center gap-8 text-sm font-medium text-white/70">
        <a href="#" onClick={(e) => handleNavClick(e, 'home')} className={`hover:text-white transition-colors ${currentView === 'home' ? 'text-white' : ''}`}>Home</a>
        <a href="#problem" onClick={(e) => handleNavClick(e, 'home', 'problem')} className="hover:text-white transition-colors">The Problem</a>
        <a href="#solution" onClick={(e) => handleNavClick(e, 'home', 'solution')} className="hover:text-white transition-colors">Our Solution</a>
        <a href="#" onClick={(e) => handleNavClick(e, 'docs')} className={`hover:text-white transition-colors ${currentView === 'docs' ? 'text-white' : ''}`}>Docs</a>
      </div>
      <div className="flex items-center gap-4">
        <button
          onClick={toggleLightMode}
          className="w-10 h-10 rounded-full border border-white/10 flex items-center justify-center text-white/80 hover:bg-white/5 transition-colors"
          title="Toggle Theme"
        >
          {isLightMode ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
        </button>
        <button 
          onClick={() => { setCurrentView('profile'); window.scrollTo({ top: 0, behavior: 'smooth' }); }}
          className={`w-10 h-10 rounded-full border flex items-center justify-center transition-colors ${currentView === 'profile' ? 'bg-primary/20 border-primary text-primary' : 'bg-surface border-white/10 text-white/80 hover:bg-white/5'}`}
          title="Driver Profile"
        >
          <User className="w-5 h-5" />
        </button>
        <button
          onClick={() => { setCurrentView('login'); window.scrollTo({ top: 0, behavior: 'smooth' }); }}
          className="hidden md:flex items-center gap-2 px-4 py-2 rounded-full bg-primary text-background font-medium hover:bg-primary/90 transition-colors"
        >
          <LogIn className="w-4 h-4" />
          <span>Login</span>
        </button>
      </div>
    </nav>
  );
};

export default Navbar;
