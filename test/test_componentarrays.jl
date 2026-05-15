const SMvN = LLPF.SimpleMvNormal

# Build a small 2-state system with named axes (position, velocity)
# and verify that ComponentVector state / ComponentMatrix covariance
# survive construction, predict! and correct! for KF and EKF.

@testset "ComponentArrays: KF round-trip" begin
    Random.seed!(0)

    μ0 = ComponentVector(position=0.0, velocity=0.0)
    ax = only(getaxes(μ0))
    Σ0 = ComponentMatrix(Matrix(1.0 * I, 2, 2), ax, ax)
    d0 = SMvN(μ0, Σ0)

    A = ComponentMatrix([1.0 0.1; 0.0 1.0], ax, ax)
    B = ComponentMatrix(reshape([0.0, 1.0], 2, 1), ax, FlatAxis())
    C = ComponentMatrix([1.0 0.0], FlatAxis(), ax)
    R1 = ComponentMatrix(Matrix(0.01 * I, 2, 2), ax, ax)
    R2 = ComponentMatrix(Matrix(0.1 * I, 1, 1), FlatAxis(), FlatAxis())

    kf = KalmanFilter(A, B, C, 0, R1, R2, d0)

    @test kf.x isa ComponentVector
    @test kf.R isa ComponentMatrix
    @test kf.x.position == 0.0
    @test kf.x.velocity == 0.0

    # One predict! step
    u = [1.0]
    predict!(kf, u)
    @test kf.x isa ComponentVector
    @test kf.R isa ComponentMatrix

    # One correct! step
    y = [0.05]
    correct!(kf, u, y)
    @test kf.x isa ComponentVector
    @test kf.R isa ComponentMatrix

    # Multi-step trajectory
    reset!(kf)
    @test kf.x isa ComponentVector
    @test kf.R isa ComponentMatrix
    for _ in 1:5
        predict!(kf, u)
        correct!(kf, u, y)
        @test kf.x isa ComponentVector
        @test kf.R isa ComponentMatrix
    end

    # Named indexing still works
    @test kf.x.position isa Real
    @test kf.x.velocity isa Real
end

@testset "ComponentArrays: EKF round-trip" begin
    Random.seed!(0)

    μ0 = ComponentVector(position=0.0, velocity=0.0)
    ax = only(getaxes(μ0))
    Σ0 = ComponentMatrix(Matrix(1.0 * I, 2, 2), ax, ax)
    d0 = SMvN(μ0, Σ0)

    # Nonlinear dynamics with named state access (a contrived nonlinearity)
    dyn(x, u, p, t) = ComponentVector(
        position = x.position + 0.1 * x.velocity,
        velocity = x.velocity - 0.01 * sin(x.position) + u[1],
    )
    meas(x, u, p, t) = [x.position]

    # User-supplied Jacobians that rewrap the ForwardDiff result as ComponentMatrix
    Ajac = function (x, u, p, t)
        J = ForwardDiff.jacobian(z -> dyn(ComponentVector(z, only(getaxes(x))), u, p, t), Vector(x))
        ComponentMatrix(J, ax, ax)
    end
    Cjac = function (x, u, p, t)
        J = ForwardDiff.jacobian(z -> meas(ComponentVector(z, only(getaxes(x))), u, p, t), Vector(x))
        ComponentMatrix(J, FlatAxis(), ax)
    end

    R1 = ComponentMatrix(Matrix(0.01 * I, 2, 2), ax, ax)
    R2 = ComponentMatrix(Matrix(0.1 * I, 1, 1), FlatAxis(), FlatAxis())

    ekf = ExtendedKalmanFilter(dyn, meas, R1, R2, d0; nu=1, ny=1, Ajac, Cjac)

    @test ekf.x isa ComponentVector
    @test ekf.R isa ComponentMatrix

    u = [0.5]
    predict!(ekf, u)
    @test ekf.x isa ComponentVector
    @test ekf.R isa ComponentMatrix

    y = [0.05]
    correct!(ekf, u, y)
    @test ekf.x isa ComponentVector
    @test ekf.R isa ComponentMatrix

    for _ in 1:5
        predict!(ekf, u)
        correct!(ekf, u, y)
        @test ekf.x isa ComponentVector
        @test ekf.R isa ComponentMatrix
    end

    @test ekf.x.position isa Real
    @test ekf.x.velocity isa Real
end

@testset "ComponentArrays: SqKF round-trip" begin
    Random.seed!(0)

    μ0 = ComponentVector(position=0.0, velocity=0.0)
    ax = only(getaxes(μ0))
    Σ0 = ComponentMatrix(Matrix(1.0 * I, 2, 2), ax, ax)
    d0 = SMvN(μ0, Σ0)

    A = ComponentMatrix([1.0 0.1; 0.0 1.0], ax, ax)
    B = ComponentMatrix(reshape([0.0, 1.0], 2, 1), ax, FlatAxis())
    C = ComponentMatrix([1.0 0.0], FlatAxis(), ax)
    R1 = ComponentMatrix(Matrix(0.01 * I, 2, 2), ax, ax)
    R2 = ComponentMatrix(Matrix(0.1 * I, 1, 1), FlatAxis(), FlatAxis())

    kf = SqKalmanFilter(A, B, C, 0, R1, R2, d0)

    @test kf.x isa ComponentVector
    @test kf.x.position == 0.0
    @test kf.x.velocity == 0.0

    u = [1.0]
    predict!(kf, u)
    @test kf.x isa ComponentVector

    y = [0.05]
    correct!(kf, u, y)
    @test kf.x isa ComponentVector

    reset!(kf)
    @test kf.x isa ComponentVector
    for _ in 1:5
        predict!(kf, u)
        correct!(kf, u, y)
        @test kf.x isa ComponentVector
    end

    @test kf.x.position isa Real
    @test kf.x.velocity isa Real
end

@testset "ComponentArrays: SqEKF round-trip" begin
    Random.seed!(0)

    μ0 = ComponentVector(position=0.0, velocity=0.0)
    ax = only(getaxes(μ0))
    Σ0 = ComponentMatrix(Matrix(1.0 * I, 2, 2), ax, ax)
    d0 = SMvN(μ0, Σ0)

    dyn(x, u, p, t) = ComponentVector(
        position = x.position + 0.1 * x.velocity,
        velocity = x.velocity - 0.01 * sin(x.position) + u[1],
    )
    meas(x, u, p, t) = [x.position]

    Ajac = function (x, u, p, t)
        J = ForwardDiff.jacobian(z -> dyn(ComponentVector(z, only(getaxes(x))), u, p, t), Vector(x))
        ComponentMatrix(J, ax, ax)
    end
    Cjac = function (x, u, p, t)
        J = ForwardDiff.jacobian(z -> meas(ComponentVector(z, only(getaxes(x))), u, p, t), Vector(x))
        ComponentMatrix(J, FlatAxis(), ax)
    end

    R1 = ComponentMatrix(Matrix(0.01 * I, 2, 2), ax, ax)
    R2 = ComponentMatrix(Matrix(0.1 * I, 1, 1), FlatAxis(), FlatAxis())

    sekf = LLPF.SqExtendedKalmanFilter(dyn, meas, R1, R2, d0; nu=1, ny=1, Ajac, Cjac)

    @test sekf.x isa ComponentVector

    u = [0.5]
    predict!(sekf, u)
    @test sekf.x isa ComponentVector

    y = [0.05]
    correct!(sekf, u, y)
    @test sekf.x isa ComponentVector

    for _ in 1:5
        predict!(sekf, u)
        correct!(sekf, u, y)
        @test sekf.x isa ComponentVector
    end

    @test sekf.x.position isa Real
    @test sekf.x.velocity isa Real
end
