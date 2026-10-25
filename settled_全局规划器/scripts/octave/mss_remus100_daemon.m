% mss_remus100_daemon.m
%
% Minimal line-oriented bridge to the MSS REMUS 100 plant. It deliberately
% contains no planner, guidance or control logic and does not modify MSS.

mss_root = getenv('MSS_ROOT');
if isempty(mss_root)
    error('MSS_ROOT is not set');
end
addpath(fileparts(mfilename('fullpath')));
addpath(genpath(mss_root));

state = zeros(12, 1);
u_actual = zeros(3, 1);
fin_limit = deg2rad(15.0);
rpm_limit = 1525.0;
fin_time_constant = 0.1;
rpm_time_constant = 1.0;
integrator = 'rk4';

fprintf(stdout, 'READY\n');
fflush(stdout);

while true
    try
        line = input('', 's');
    catch
        break;
    end
    line = strtrim(line);
    if isempty(line)
        continue;
    end
    tokens = strsplit(line);
    command = tokens{1};

    switch command
        case 'INIT'
            values = cellfun(@str2double, tokens(2:end));
            if numel(values) ~= 12 || any(~isfinite(values))
                error('INIT requires 12 finite values');
            end
            state = values(:);
            u_actual = zeros(3, 1);
            fprintf(stdout, 'OK\n');
            fflush(stdout);

        case 'ACT'
            values = cellfun(@str2double, tokens(2:end));
            if numel(values) ~= 3 || any(~isfinite(values))
                error('ACT requires 3 finite values');
            end
            u_actual = values(:);
            u_actual(1) = sat(u_actual(1), fin_limit);
            u_actual(2) = sat(u_actual(2), fin_limit);
            u_actual(3) = sat(u_actual(3), rpm_limit);
            fprintf(stdout, 'OK\n');
            fflush(stdout);

        case 'INTEG'
            if numel(tokens) ~= 2 || ~any(strcmp(tokens{2}, {'rk4', 'euler'}))
                error('INTEG requires rk4 or euler');
            end
            integrator = tokens{2};
            fprintf(stdout, 'OK\n');
            fflush(stdout);

        case 'DERIV'
            values = cellfun(@str2double, tokens(2:end));
            if numel(values) ~= 3 || any(~isfinite(values))
                error('DERIV requires Vc betaVc w_c');
            end
            [xdot, U] = remus100(state, u_actual, values(1), values(2), values(3));
            write_result(xdot, U, u_actual);

        case 'MASS'
            [~, ~, M] = remus100();
            values = M(1:7:36);
            fprintf(stdout, 'MASS %.17g %.17g %.17g %.17g %.17g %.17g\n', values);
            fflush(stdout);

        case 'STEP'
            values = cellfun(@str2double, tokens(2:end));
            if numel(values) ~= 7 || any(~isfinite(values))
                error('STEP requires delta_r delta_s rpm Vc betaVc w_c dt');
            end
            command_actual = values(1:3);
            command_actual(1) = sat(command_actual(1), fin_limit);
            command_actual(2) = sat(command_actual(2), fin_limit);
            command_actual(3) = sat(command_actual(3), rpm_limit);
            Vc = values(4);
            betaVc = values(5);
            w_c = values(6);
            dt = values(7);
            if dt <= 0 || ~isfinite(dt)
                error('STEP dt must be finite and positive');
            end

            if strcmp(integrator, 'rk4')
                state = rk4(@remus100, dt, state, u_actual, Vc, betaVc, w_c);
            else
                state = state + dt * remus100(state, u_actual, Vc, betaVc, w_c);
            end
            u_actual(1) = u_actual(1) + dt * (command_actual(1) - u_actual(1)) / fin_time_constant;
            u_actual(2) = u_actual(2) + dt * (command_actual(2) - u_actual(2)) / fin_time_constant;
            u_actual(3) = u_actual(3) + dt * (command_actual(3) - u_actual(3)) / rpm_time_constant;
            u_actual(1) = sat(u_actual(1), fin_limit);
            u_actual(2) = sat(u_actual(2), fin_limit);
            u_actual(3) = sat(u_actual(3), rpm_limit);

            U = sqrt(state(1)^2 + state(2)^2 + state(3)^2);
            write_result(state, U, u_actual);

        case 'QUIT'
            break;

        otherwise
            error('Unknown command: %s', command);
    end
end
