function write_result(state_or_derivative, U, u_actual)
    values = [state_or_derivative(:); U; u_actual(:)];
    fprintf(stdout, 'RESULT %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n', values);
    fflush(stdout);
end
