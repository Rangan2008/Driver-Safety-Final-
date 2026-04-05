import React from 'react';
import { Shield } from 'lucide-react';

const Footer = () => (
  <footer className="py-12 border-t border-white/10 bg-background text-center text-white/40 text-sm">
    <div className="container mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-4">
      <div className="flex items-center gap-2">
        <Shield className="w-5 h-5 text-primary/50" />
        <span className="font-bold font-display text-white/60">DriveSafe</span>
      </div>
      <p>&copy; 2026 DriveSafe Technologies. All rights reserved.</p>
      <div className="flex gap-6">
        <a href="#" className="hover:text-white transition-colors">Privacy</a>
        <a href="#" className="hover:text-white transition-colors">Terms</a>
      </div>
    </div>
  </footer>
);

export default Footer;
