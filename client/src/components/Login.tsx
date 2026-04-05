import React from 'react';
import { Shield, Mail, Lock } from 'lucide-react';

const Login = ({ setCurrentView }: { setCurrentView: (v: 'home' | 'profile' | 'showcase' | 'docs' | 'login') => void }) => {
  return (
    <div className="min-h-screen flex items-center justify-center pt-20 pb-12 px-6">
      <div className="w-full max-w-md bg-surface border border-white/10 rounded-2xl p-8 shadow-2xl relative overflow-hidden">
        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-primary via-alert to-primary" />
        <div className="text-center mb-8">
          <Shield className="w-12 h-12 text-primary mx-auto mb-4" />
          <h2 className="text-3xl font-bold font-display tracking-tight">Welcome Back</h2>
          <p className="text-white/60 mt-2">Sign in to your DriveSafe account</p>
        </div>
        
        <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); setCurrentView('home'); }}>
          <div>
            <label className="block text-sm font-medium text-white/70 mb-1">Email Address</label>
            <div className="relative">
              <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-white/40" />
              <input 
                type="email" 
                required
                className="w-full bg-background border border-white/10 rounded-lg pl-10 pr-4 py-3 text-white focus:outline-none focus:border-primary transition-colors" 
                placeholder="you@example.com" 
              />
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-white/70 mb-1">Password</label>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-white/40" />
              <input 
                type="password" 
                required
                className="w-full bg-background border border-white/10 rounded-lg pl-10 pr-4 py-3 text-white focus:outline-none focus:border-primary transition-colors" 
                placeholder="••••••••" 
              />
            </div>
          </div>
          
          <div className="flex items-center justify-between text-sm">
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" className="rounded border-white/20 bg-background text-primary focus:ring-primary" />
              <span className="text-white/60">Remember me</span>
            </label>
            <a href="#" className="text-primary hover:underline">Forgot password?</a>
          </div>
          
          <button 
            type="submit"
            className="w-full bg-primary text-background font-semibold py-3 rounded-lg hover:bg-primary/90 transition-colors mt-6"
          >
            Sign In
          </button>
        </form>
        
        <div className="mt-8 pt-6 border-t border-white/10 text-center">
          <p className="text-white/60 text-sm">
            Don't have an account? <a href="#" className="text-primary hover:underline font-medium">Sign up</a>
          </p>
        </div>
      </div>
    </div>
  );
};

export default Login;
