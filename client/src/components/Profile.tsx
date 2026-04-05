import React from 'react';
import { motion } from 'motion/react';
import { User, Car, HeartPulse, Phone, Shield } from 'lucide-react';

const Profile = () => {
  return (
    <div className="pt-32 pb-24 container mx-auto px-6 max-w-4xl">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-10"
      >
        <h1 className="text-4xl font-bold mb-4">Driver Profile</h1>
        <p className="text-white/60 text-lg">
          Securely store your medical and vehicle information. This data is encrypted and only accessed by emergency responders during a critical event.
        </p>
      </motion.div>
      
      <div className="space-y-8">
        {/* Personal Details */}
        <motion.section 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-surface border border-white/10 rounded-2xl p-8"
        >
          <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
            <User className="w-6 h-6 text-primary"/> Personal Details
          </h2>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label className="text-sm text-white/60">Full Name</label>
              <input type="text" className="w-full bg-background border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-primary transition-colors" placeholder="John Doe" />
            </div>
            <div className="space-y-2">
              <label className="text-sm text-white/60">Date of Birth</label>
              <input type="date" className="w-full bg-background border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-primary transition-colors" />
            </div>
            <div className="space-y-2">
              <label className="text-sm text-white/60">Driver's License Number</label>
              <input type="text" className="w-full bg-background border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-primary transition-colors" placeholder="DL-12345678" />
            </div>
            <div className="space-y-2">
              <label className="text-sm text-white/60">Home Address</label>
              <input type="text" className="w-full bg-background border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-primary transition-colors" placeholder="123 Safety Ave, Tech City" />
            </div>
          </div>
        </motion.section>
        
        {/* Vehicle Details */}
        <motion.section 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-surface border border-white/10 rounded-2xl p-8"
        >
          <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
            <Car className="w-6 h-6 text-primary"/> Vehicle Information
          </h2>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label className="text-sm text-white/60">Vehicle Make</label>
              <input type="text" className="w-full bg-background border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-primary transition-colors" placeholder="e.g. Toyota, Tesla" />
            </div>
            <div className="space-y-2">
              <label className="text-sm text-white/60">Vehicle Model</label>
              <input type="text" className="w-full bg-background border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-primary transition-colors" placeholder="e.g. Camry, Model 3" />
            </div>
            <div className="space-y-2">
              <label className="text-sm text-white/60">Year</label>
              <input type="text" className="w-full bg-background border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-primary transition-colors" placeholder="2024" />
            </div>
            <div className="space-y-2">
              <label className="text-sm text-white/60">License Plate</label>
              <input type="text" className="w-full bg-background border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-primary transition-colors" placeholder="ABC-1234" />
            </div>
          </div>
        </motion.section>

        {/* Medical Details */}
        <motion.section 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-surface border border-white/10 rounded-2xl p-8"
        >
          <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
            <HeartPulse className="w-6 h-6 text-alert"/> Medical Profile
          </h2>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label className="text-sm text-white/60">Blood Type</label>
              <select className="w-full bg-background border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-primary transition-colors appearance-none">
                <option value="">Select Blood Type</option>
                <option value="A+">A+</option>
                <option value="A-">A-</option>
                <option value="B+">B+</option>
                <option value="B-">B-</option>
                <option value="O+">O+</option>
                <option value="O-">O-</option>
                <option value="AB+">AB+</option>
                <option value="AB-">AB-</option>
              </select>
            </div>
            <div className="space-y-2">
              <label className="text-sm text-white/60">Pre-existing Conditions</label>
              <input type="text" className="w-full bg-background border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-primary transition-colors" placeholder="e.g. Asthma, Diabetes" />
            </div>
            <div className="space-y-2 md:col-span-2">
              <label className="text-sm text-white/60">Allergies</label>
              <input type="text" className="w-full bg-background border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-primary transition-colors" placeholder="e.g. Penicillin, Peanuts" />
            </div>
            <div className="space-y-2 md:col-span-2">
              <label className="text-sm text-white/60">Current Medications</label>
              <textarea className="w-full bg-background border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-primary transition-colors min-h-[100px]" placeholder="List any medications you are currently taking..."></textarea>
            </div>
          </div>
        </motion.section>

        {/* Emergency Contacts */}
        <motion.section 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-surface border border-white/10 rounded-2xl p-8"
        >
          <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
            <Phone className="w-6 h-6 text-success"/> Emergency Contacts
          </h2>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label className="text-sm text-white/60">Primary Contact Name</label>
              <input type="text" className="w-full bg-background border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-primary transition-colors" placeholder="Jane Doe" />
            </div>
            <div className="space-y-2">
              <label className="text-sm text-white/60">Relationship</label>
              <input type="text" className="w-full bg-background border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-primary transition-colors" placeholder="Spouse" />
            </div>
            <div className="space-y-2 md:col-span-2">
              <label className="text-sm text-white/60">Phone Number</label>
              <input type="tel" className="w-full bg-background border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-primary transition-colors" placeholder="+1 (555) 000-0000" />
            </div>
          </div>
        </motion.section>

        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="flex justify-end pt-4"
        >
          <button className="px-10 py-4 bg-primary text-background font-bold text-lg rounded-full hover:bg-primary/90 transition-transform hover:scale-105 flex items-center gap-2">
            <Shield className="w-5 h-5" /> Save Profile
          </button>
        </motion.div>
      </div>
    </div>
  );
};

export default Profile;
