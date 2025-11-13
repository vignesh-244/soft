import { createContext, useContext, useEffect, useState, ReactNode } from 'react';

interface User {
  id: string;
  email: string;
  user_metadata: { name: string };
  app_metadata: {};
  aud: string;
  created_at: string;
  updated_at: string;
  role: string;
  confirmation_sent_at: string;
  recovery_sent_at: string;
  email_change_sent_at: string;
  new_email: string;
  invited_at: string;
  action_link: string;
  email_confirmed_at: string;
  phone_confirmed_at: string;
  confirmed_at: string;
  last_sign_in_at: string;
  phone: string;
  factors: any[];
}

interface AuthError {
  message: string;
}

interface AuthContextType {
  user: User | null;
  loading: boolean;
  signUp: (email: string, password: string) => Promise<{ error: AuthError | null }>;
  signIn: (email: string, password: string) => Promise<{ error: AuthError | null }>;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// Mock user for demo purposes
const mockUser: User = {
  id: 'demo-user-id',
  email: 'vigneshnarayanan2003@gmail.com',
  user_metadata: { name: 'Demo User' },
  app_metadata: {},
  aud: 'authenticated',
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  role: 'authenticated',
  confirmation_sent_at: new Date().toISOString(),
  recovery_sent_at: new Date().toISOString(),
  email_change_sent_at: new Date().toISOString(),
  new_email: 'vigneshnarayanan2003@gmail.com',
  invited_at: new Date().toISOString(),
  action_link: '',
  email_confirmed_at: new Date().toISOString(),
  phone_confirmed_at: new Date().toISOString(),
  confirmed_at: new Date().toISOString(),
  last_sign_in_at: new Date().toISOString(),
  phone: '',
  factors: []
};

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if user was previously logged in (stored in localStorage)
    const storedUser = localStorage.getItem('demo-user');
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    }
    setLoading(false);
  }, []);

  const signUp = async (email: string, password: string) => {
    // Simulate successful signup for demo
    console.log('Demo signup successful for:', email);
    return { error: null };
  };

  const signIn = async (email: string, password: string) => {
    // Check demo credentials
    if (email === 'vigneshnarayanan2003@gmail.com' && password === 'admin') {
      console.log('Demo login successful');
      setUser(mockUser);
      localStorage.setItem('demo-user', JSON.stringify(mockUser));
      return { error: null };
    } else {
      return { error: { message: 'Invalid credentials. Use demo credentials.' } as AuthError };
    }
  };

  const signOut = async () => {
    setUser(null);
    localStorage.removeItem('demo-user');
    console.log('Demo logout successful');
  };

  return (
    <AuthContext.Provider value={{ user, loading, signUp, signIn, signOut }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
