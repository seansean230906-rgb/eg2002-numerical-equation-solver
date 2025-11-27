import streamlit as st
import math
import numpy as np
from typing import Callable, List, Tuple, Optional
import io

def evaluate_equation(equation_func: Callable, x: float) -> Optional[float]:
    """Safely evaluate equation at point x, return None on error."""
    try:
        result = equation_func(x)
        # Also check for NaN values
        if result is None or math.isnan(result):
            return None
        return result
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as e:
        return None

def successive_bisection(equation: Callable, a: float, b: float, tolerance: float, max_iter: int) -> Tuple[Optional[float], int, bool, str]:
    """Successive Bisection Method to find root in interval [a,b]."""
    
    fa = evaluate_equation(equation, a)
    fb = evaluate_equation(equation, b)
    
    # Check if initial evaluations are valid
    if fa is None or fb is None:
        return None, 0, False, "Function evaluation failed."
    
    # Check if root exists in interval
    if fa * fb > 0:
        return None, 0, False, "No sign change in interval."
    
    iterations = 0
    for i in range(max_iter):
        iterations += 1
        c = (a + b) / 2
        fc = evaluate_equation(equation, c)
        
        if fc is None:
            return None, iterations, False, "Function evaluation failed during bisection."
            
        if abs(fc) < tolerance or (b - a) / 2 < tolerance:
            return c, iterations, True, "Converged successfully."
            
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    
    return None, iterations, False, f"Max iterations ({max_iter}) reached without convergence."

def find_all_roots(equation: Callable, x_min: float, x_max: float, step_size: float, tolerance: float, max_iter: int) -> Tuple[List[float], List[str]]:
    """Find all roots of equation in range [x_min, x_max]."""
    roots = []
    status_messages = []
    processed_roots = set()  # Avoid duplicates
    convergence_failures = 0
    total_intervals_checked = 0
    
    x = x_min
    while x < x_max:
        total_intervals_checked += 1
        a = x
        b = min(x + step_size, x_max)
        
        # Evaluate function at interval endpoints
        fa = evaluate_equation(equation, a)
        fb = evaluate_equation(equation, b)
        
        # Skip interval if function evaluation failed
        if fa is None or fb is None:
            # Don't add warning messages for evaluation failures
            x = b
            continue
            
        # Skip if not numeric types
        if not isinstance(fa, (int, float)) or not isinstance(fb, (int, float)):
            # Don't add warning messages for non-numeric results
            x = b
            continue
        
        # Check for sign change (root in interval)
        if fa * fb <= 0:
            root, iterations, success, convergence_message = successive_bisection(equation, a, b, tolerance, max_iter)
            
            if success:
                # VERIFY the root actually meets tolerance with reasonable margin
                f_root = evaluate_equation(equation, root)
                if f_root is not None and abs(f_root) < tolerance * 10:  # Allow 10x tolerance margin
                    rounded_root = round(root, 8)
                    if rounded_root not in processed_roots:
                        roots.append(root)
                        processed_roots.add(rounded_root)
                        status_messages.append(f"✓ Root found at x = {root:.6f} (after {iterations} iterations).")
                else:
                    status_messages.append(f"✗ Root at x = {root:.6f} failed verification: f(x) = {f_root:.2e}.")
            else:
                convergence_failures += 1
                # Only add convergence failure messages (not evaluation failures)
                if "function evaluation failed" not in convergence_message.lower():
                    status_messages.append(f"✗ Failed to converge in [{a:.3f}, {b:.3f}]: {convergence_message}")
        
        x = b
    
    # Add summary message about convergence issues
    if not roots and convergence_failures > 0:
        status_messages.append(f"⚠️ Note: Found {convergence_failures} interval(s) with potential roots that failed to converge.")
        status_messages.append(f"💡 Try increasing Max Iterations (current: {max_iter}) or Step Size for better convergence.")
    elif not roots:
        status_messages.append("No roots found in the given range.")
    else:
        status_messages.append(f"Search complete: checked {total_intervals_checked} intervals, found {len(roots)} root(s).")
    
    return roots, status_messages

def create_equation(a: float, w: float, power: float, phase: float, c: float) -> Callable:
    """Create the equation function: a * x^power = e^(c*x) * sin(w*x + phase)."""
    def equation(x: float) -> float:
        try:
            # Handle x = 0 with negative power (division by zero)
            if x == 0 and power < 0:
                return float('nan')  # Undefined at x=0 when power is negative
            
            # Handle negative x with negative OR non-integer power
            if x < 0 and (power < 0 or not power.is_integer()):
                return float('nan')  # Complex result or undefined
            
            # For valid cases, compute normally
            left_side = a * (x ** power)
            right_side = math.exp(c * x) * math.sin(w * x + phase)
            return left_side - right_side  # f(x) = 0 form
            
        except (ValueError, ZeroDivisionError):
            return float('nan')
    
    return equation

def validate_solver_settings(x_min: float, x_max: float, step_size: float) -> Tuple[bool, str]:
    """Validate solver settings and return (is_valid, error_message)."""
    if x_min >= x_max:
        return False, f"❌ **Invalid Search Range**: Min. x ({x_min}) must be less than Max. x ({x_max})."
    
    if step_size <= 0:
        return False, f"❌ **Invalid Step Size**: Step size ({step_size}) must be positive."
    
    if step_size > (x_max - x_min):
        return False, f"❌ **Step Size Too Large**: Step size ({step_size}) is larger than search range ({x_max - x_min:.6f})."
    
    return True, ""

def plot_function(equation_func: Callable, x_min: float, x_max: float, roots: List[float], equation_text: str):
    """Plot the function and any roots found."""
    try:
        import matplotlib.pyplot as plt
        
        # Only plot valid x values where function is defined
        x_vals = np.linspace(x_min, x_max, 1000)
        y_vals = []
        valid_x_vals = []
        
        for x_val in x_vals:
            y_val = evaluate_equation(equation_func, x_val)
            if y_val is not None:
                valid_x_vals.append(x_val)
                y_vals.append(y_val)
        
        # Smaller figure size for better fit on screen
        fig, ax = plt.subplots(figsize=(8, 4))  # Reduced from (10, 6)
        ax.plot(valid_x_vals, y_vals, 'b-', label='f(x)', linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        # Plot roots if any were found
        if roots:
            root_vals = [evaluate_equation(equation_func, root) for root in roots]
            ax.plot(roots, root_vals, 'ro', markersize=6, label='Roots')
        
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(f'Function Plot: {equation_text}', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        st.warning(f"Could not generate plot: {e}")
        return None

def save_plot_as_png(fig):
    """Save matplotlib figure as PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf

def save_plot_as_svg(fig):
    """Save matplotlib figure as SVG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    buf.seek(0)
    return buf

def save_plot_as_pdf(fig):
    """Save matplotlib figure as PDF bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', bbox_inches='tight')
    buf.seek(0)
    return buf

def main():
    st.set_page_config(page_title="Numerical Equation Solver", layout="wide")
    
    # Add custom CSS for tooltips
    st.markdown("""
    <style>
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        margin-left: 8px;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 280px;
        background-color: #333;
        color: white;
        text-align: left;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1000;
        top: -10px;
        left: 30px;
        font-size: 0.8em;
        line-height: 1.4;
        opacity: 0;
        transition: opacity 0.3s;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .tooltip .tooltiptext::after {
        content: "";
        position: absolute;
        top: 15px;
        right: 100%;
        margin-top: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: transparent #333 transparent transparent;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    .param-header {
        display: flex;
        align-items: center;
        margin-bottom: 5px;
    }
    .tooltip-icon {
        color: #666;
        font-size: 0.9em;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Gradient header with Option 1 styling
    st.markdown("""
    <h1 style='
        text-align: center; 
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        margin-bottom: 10px;
    '>Numerical Equation Solver</h1>
    """, unsafe_allow_html=True)
    
    # Subheadings with matching style
    st.markdown("""
    <h3 style='
        text-align: center; 
        color: #2c3e50;
        margin: 10px 0 5px 0;
        font-weight: 500;
    '>Solve equations of the form: axᵇ = eᶜˣsin(wx + v)</h3>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <h4 style='
        text-align: center; 
        color: #7f8c8d;
        margin: 0 0 20px 0;
        font-weight: 400;
    '>Using the method of successive bisections</h4>
    """, unsafe_allow_html=True)
    
    # QUICK GUIDE SECTION - SIMPLE ADDITION
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>📚 Quick Guide</h2>", unsafe_allow_html=True)
    
    guide_col1, guide_col2 = st.columns(2)
    
    with guide_col1:
        st.markdown("**📝 How to Use:**")
        st.markdown("""
        1. **Set Parameters**: Adjust a, w, b, v, c for your equation.
        2. **Configure Solver**: Set search range and precision settings.
        3. **Find Roots**: Click the 'Find All Roots' button.
        4. **Analyze**: View results, plot, and download data.
        """)
        
    with guide_col2:
        st.markdown("**💡 Tips:**")
        st.markdown("""
        • Use smaller step sizes for more roots.  
        • Increase search range if no roots found.  
        • Higher iterations help with difficult equations.  
        • Hover over ❓ icons for parameter help.  
        """)
    
    st.markdown("---")
    
    # Create two columns for parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h2 style='text-align: center; background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>🔧 Equation Parameters</h2>", unsafe_allow_html=True)
        
        # Equation parameters with tooltips
        st.markdown("""
        <div class="param-header">
            <strong style="font-size: 1.1em;">a</strong>
            <div class="tooltip">
                <span class="tooltip-icon">❓</span>
                <span class="tooltiptext">Coefficient 'a' scales the polynomial term axᵇ.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        a = st.number_input("a", value=0.50, step=0.1, format="%.6f", key="a", label_visibility="collapsed")
        
        st.markdown("""
        <div class="param-header">
            <strong style="font-size: 1.1em;">w</strong>
            <div class="tooltip">
                <span class="tooltip-icon">❓</span>
                <span class="tooltiptext">Angular frequency 'w' controls the oscillation frequency in sin(wx + v).</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        w = st.number_input("w", value=4.20, step=0.1, format="%.6f", key="w", label_visibility="collapsed")
        
        st.markdown("""
        <div class="param-header">
            <strong style="font-size: 1.1em;">b</strong>
            <div class="tooltip">
                <span class="tooltip-icon">❓</span>
                <span class="tooltiptext">Power 'b' is the exponent of x in axᵇ. Use caution with negative values and non-integer powers when x < 0.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        power = st.number_input("b", value=-1.40, step=0.1, format="%.6f", key="power", label_visibility="collapsed")
        
        st.markdown("""
        <div class="param-header">
            <strong style="font-size: 1.1em;">v</strong>
            <div class="tooltip">
                <span class="tooltip-icon">❓</span>
                <span class="tooltiptext">Phase shift 'v' provides a horizontal shift of the sine wave in sin(wx + v).</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        phase = st.number_input("v", value=0.20, step=0.1, format="%.6f", key="phase", label_visibility="collapsed")
        
        st.markdown("""
        <div class="param-header">
            <strong style="font-size: 1.1em;">c</strong>
            <div class="tooltip">
                <span class="tooltip-icon">❓</span>
                <span class="tooltiptext">Exponential coefficient 'c' controls the growth (c > 0) or decay (c < 0) rate in e^(cx).</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        c = st.number_input("c", value=0.10, step=0.1, format="%.6f", key="c", label_visibility="collapsed")
    
    with col2:
        st.markdown("<h2 style='text-align: center; background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>🎯 Solver Parameters</h2>", unsafe_allow_html=True)
        
        # Solver parameters with tooltips
        st.markdown("""
        <div class="param-header">
            <strong style="font-size: 1.1em;">Min. x</strong>
            <div class="tooltip">
                <span class="tooltip-icon">❓</span>
                <span class="tooltiptext">Minimum x-value for root search. The solver scans from Min. x to Max. x.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        x_min = st.number_input("Min. x", value=-5.00, step=0.5, format="%.6f", key="x_min", label_visibility="collapsed")
        
        st.markdown("""
        <div class="param-header">
            <strong style="font-size: 1.1em;">Max. x</strong>
            <div class="tooltip">
                <span class="tooltip-icon">❓</span>
                <span class="tooltiptext">Maximum x-value for root search. The solver scans from Min. x to Max. x.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        x_max = st.number_input("Max. x", value=5.00, step=0.5, format="%.6f", key="x_max", label_visibility="collapsed")
        
        st.markdown("""
        <div class="param-header">
            <strong style="font-size: 1.1em;">Step Size</strong>
            <div class="tooltip">
                <span class="tooltip-icon">❓</span>
                <span class="tooltiptext">Step size for interval scanning. Smaller values provide more precision but are slower. Larger values are faster but may miss roots.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        step_size = st.number_input("Step Size", value=0.01, step=0.01, format="%.6f", key="step_size", label_visibility="collapsed")
        
        st.markdown("""
        <div class="param-header">
            <strong style="font-size: 1.1em;">Tolerance</strong>
            <div class="tooltip">
                <span class="tooltip-icon">❓</span>
                <span class="tooltiptext">Convergence tolerance. Smaller values yield more accurate roots but may require more iterations.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        tolerance = st.number_input("Tolerance", value=1e-8, format="%.e", key="tolerance", label_visibility="collapsed")
        
        st.markdown("""
        <div class="param-header">
            <strong style="font-size: 1.1em;">Max Iterations</strong>
            <div class="tooltip">
                <span class="tooltip-icon">❓</span>
                <span class="tooltiptext">Maximum iterations per root. Higher values help with difficult convergence but increase computation time.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        max_iter = st.number_input("Max Iterations", value=200, step=50, key="max_iter", label_visibility="collapsed")
    
    st.markdown("---")
    
    # Display the current equation with label
    st.markdown("<h3 style='text-align: center; color: #2c3e50; margin: 20px 0 10px 0;'>📐 Current Equation</h3>", unsafe_allow_html=True)
    
    equation_text = f"{a}x^({power}) = e^({c}x)sin({w}x + {phase})"
    st.latex(f"{a}x^{{{power}}} = e^{{{c}x}}\\sin({w}x + {phase})")
    
    # Solve button
    if st.button("🔍 Find All Roots", type="primary"):
        try:
            # Validate solver settings first
            is_valid, error_message = validate_solver_settings(x_min, x_max, step_size)
            if not is_valid:
                st.error(error_message)
                st.info("🔧 **Please adjust your solver settings and try again.**")
                return
            
            # Create equation function
            equation_func = create_equation(a, w, power, phase, c)
                
            # Find roots
            with st.spinner("🔍 Searching for roots..."):
                roots, status_messages = find_all_roots(equation_func, x_min, x_max, step_size, tolerance, max_iter)
            
            # Display results
            st.markdown("<h2 style='text-align: center; background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>📊 Results</h2>", unsafe_allow_html=True)
            
            if roots:
                st.success(f"🎉 Found {len(roots)} root(s):")
                for i, root in enumerate(roots, 1):
                    st.write(f"📍 Root {i}: x = {root:.8f}")
                    
                    # Verify the root
                    verification = evaluate_equation(equation_func, root)
                    if verification is not None:
                        st.write(f"✅ Verification: f({root:.6f}) = {verification:.2e}")
            
            # ALWAYS SHOW DETAILED SOLVER LOG (but without evaluation failure messages)
            with st.expander("📋 Detailed Solver Log"):
                if status_messages:
                    for msg in status_messages:
                        st.write(msg)
                else:
                    st.write("No significant solver events to report.")
            
            if not roots:
                # SHOW NO ROOTS MESSAGE PROMINENTLY
                st.warning("❌ No roots found in the given search range.")
                
                # Check if there were convergence failures and provide specific advice
                convergence_failures = sum(1 for msg in status_messages if "failed to converge" in msg.lower())
                if convergence_failures > 0:
                    st.info(f"💡 **Convergence Issue Detected**: {convergence_failures} potential root(s) failed to converge.")
                    st.info(f"🔧 **Try**: Increasing 'Max Iterations' (current: {max_iter}) or using a larger 'Step Size'.")
                else:
                    st.info("💡 Try adjusting the search range or equation parameters.")
            
            # ALWAYS PLOT THE FUNCTION (with or without roots)
            st.markdown("<h2 style='text-align: center; background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>📈 Function Plot</h2>", unsafe_allow_html=True)
            fig = plot_function(equation_func, x_min, x_max, roots, equation_text)
            
            if fig is not None:
                # Use container to center the plot and control its width
                with st.container():
                    st.pyplot(fig, use_container_width=False)
                
                # DOWNLOAD OPTIONS SECTION
                st.markdown("---")
                st.markdown("<h3 style='text-align: center; color: #2c3e50; margin: 20px 0 10px 0;'>💾 Download Plot</h3>", unsafe_allow_html=True)
                
                # Create columns for download buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    png_buf = save_plot_as_png(fig)
                    st.download_button(
                        label="📥 Download as PNG",
                        data=png_buf,
                        file_name=f"function_plot_{a}_{w}_{power}_{phase}_{c}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                with col2:
                    svg_buf = save_plot_as_svg(fig)
                    st.download_button(
                        label="📥 Download as SVG",
                        data=svg_buf,
                        file_name=f"function_plot_{a}_{w}_{power}_{phase}_{c}.svg",
                        mime="image/svg+xml",
                        use_container_width=True
                    )
                
                with col3:
                    pdf_buf = save_plot_as_pdf(fig)
                    st.download_button(
                        label="📥 Download as PDF",
                        data=pdf_buf,
                        file_name=f"function_plot_{a}_{w}_{power}_{phase}_{c}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                
                st.info("💡 **Tip**: PNG is best for web use, SVG for vector graphics, and PDF for printing.")
                    
        except Exception as e:
            st.error(f"🚨 Error solving equation: {str(e)}")

if __name__ == "__main__":
    main()