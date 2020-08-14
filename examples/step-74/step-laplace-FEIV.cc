/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2009 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Timo Heister and Jiaqi Zhang, Clemson University, 2020
 */


// The first few files have already been covered in previous examples and will
// thus not be further commented on:
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/mapping_q1.h>
// Here the discontinuous finite elements are defined. They are used in the
// same way as all other finite elements, though -- as you have seen in
// previous tutorial programs -- there isn't much user interaction with finite
// element classes at all: they are passed to <code>DoFHandler</code> and
// <code>FEValues</code> objects, and that is about it.
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_interface_values.h>
// We are going to use the simplest possible solver, called Richardson
// iteration, that represents a simple defect correction. This, in combination
// with a block SSOR preconditioner (defined in precondition_block.h), that
// uses the special block matrix structure of system matrices arising from DG
// discretizations.
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/precondition.h>
// We are going to use gradients as refinement indicator.
#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/convergence_table.h>

#include <deal.II/meshworker/copy_data.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/meshworker/scratch_data.h>



namespace Step74
{
  using namespace dealii;

  enum Test_Case
  {
    convergence_rate,
    l_singularity
  };


  template <int dim>
  class Solution:  public Function<dim>
  {
  public:
    Solution(Test_Case test_case)
      :
      Function<dim>(),
      test_case(test_case)
      {}
    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<double> &values,
                             const unsigned int component=0) const;
    virtual Tensor<1,dim> gradient(const Point<dim> &point,
                                   const unsigned int component=0) const;
    private:
    Functions::LSingularityFunction ref;
    Test_Case test_case;

  };

  template <int dim>
  void Solution<dim>:: value_list (const std::vector<Point<dim> > &points,
                                   std::vector<double> &values,
                                   const unsigned int component) const
  {
    switch (this->test_case)
      {
        case convergence_rate :
          {
            using numbers::PI;
            for (unsigned int i=0; i<values.size(); ++i)
              values[i]= std::sin(2*PI*points[i][0]) * std::sin(2*PI*points[i][1]);
            break;
          }
        case l_singularity :
          {
            for (unsigned int i=0; i<values.size(); ++i)
              values[i]= ref.value(points[i]);
            break;              
          }
        default:
          {
            Assert(false, ExcNotImplemented());
          }             
      }
      
  }


  template <int dim>
  Tensor<1,dim> Solution<dim>:: gradient (const Point<dim> &point,
                                 const unsigned int component) const
  {
    switch (this->test_case)
      {
        case convergence_rate :
          {
            Tensor<1,dim> return_value;
            using numbers::PI;
            return_value[0] =2*PI* std::cos(2*PI*point[0]) * std::sin(2*PI*point[1]);
            return_value[1] =2*PI* std::sin(2*PI*point[0]) * std::cos(2*PI*point[1]);
            return return_value;
            break;
          }
        case l_singularity :
          {
            return ref.gradient (point);
            break;              
          }
        default:
          {
            Assert(false, ExcNotImplemented());
          }             
      }
      
  }


  template <int dim>
  class Viscosity:  public Function<dim>
  {
  public:
    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<double> &values,
                             const unsigned int component=0) const
    {
      for (unsigned int i=0; i<values.size(); ++i)
        values[i]=1.0;
    }
  }; 


  template <int dim>
  class RHS:  public Function<dim>
  {
  public:
    RHS(Test_Case test_case)
      :
      Function<dim>(),
      test_case(test_case)
      {}
    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<double> &values,
                             const unsigned int component=0) const
        {
        switch (test_case)
          {
            case convergence_rate :
              {
                using numbers::PI;
                for (unsigned int i=0; i<values.size(); ++i)
                  values[i]= 8*PI*PI*std::sin(2*PI*points[i][0]) * std::sin(2*PI*points[i][1]);
                break;
              }
            case l_singularity :
              {
                for (unsigned int i=0; i<values.size(); ++i)
                  //assuming that viscosity nu =1
                  values[i]= ref.laplacian(points[i]);
                break;              
              }
            default:
              {
                Assert(false, ExcNotImplemented());
              }             
          }
        }
                                       

    private:
    Functions::LSingularityFunction ref;
    Test_Case test_case;
                   
  };

  // @sect3{The SIPGLaplace class}
  //
  // After this preparations, we proceed with the main class of this program,
  // called SIPGLaplace. It is basically the main class of step-6. We do
  // not have a ConstraintMatrix, because there are no hanging node
  // constraints in DG discretizations.

  // Major differences will only come up in the implementation of the assemble
  // functions, since here, we not only need to cover the flux integrals over
  // faces, we also use the MeshWorker interface to simplify the loops
  // involved.

struct CopyDataFace
{
  FullMatrix<double>                   cell_matrix;
  std::vector<types::global_dof_index> joint_dof_indices;
  double values[2];
  unsigned int cell_indices[2];
};

struct CopyData
{
  FullMatrix<double>                   cell_matrix;
  Vector<double>                       cell_rhs;
  std::vector<types::global_dof_index> local_dof_indices;
  std::vector<CopyDataFace> face_data;
  double value;
  unsigned int  cell_index; 
  template<class Iterator>
  void reinit(const Iterator &cell, unsigned int dofs_per_cell)
  {
    cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
    cell_rhs.reinit(dofs_per_cell);
    local_dof_indices.resize(dofs_per_cell);
    cell->get_dof_indices(local_dof_indices);
  }  

};

template <class MatrixType, class VectorType>
inline void copy(const CopyData &c,
          const AffineConstraints<double> &constraints,
          MatrixType & system_matrix,
          VectorType & system_rhs)
{
  constraints.distribute_local_to_global(c.cell_matrix,
                                         c.cell_rhs,
                                         c.local_dof_indices,
                                         system_matrix,
                                         system_rhs);
  for (auto &cdf : c.face_data)
    {
      const unsigned int dofs_per_cell   = cdf.joint_dof_indices.size();
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int k=0; k<dofs_per_cell; ++k)
          system_matrix.add(cdf.joint_dof_indices[i], cdf.joint_dof_indices[k],
                            cdf.cell_matrix(i,k));
    }
}


  template <int dim>
  class SIPGLaplace
  {
  public:
    SIPGLaplace ();
    void run ();

  private:
    void setup_system ();
    void assemble_system ();
    void solve (Vector<double> &solution);
    void refine_grid ();
    void output_results (const unsigned int cycle) const;
    double compute_penalty(const double h1, const double h2);
    void compute_errors();
    void compute_error_estimate();

    const Test_Case test_case =  convergence_rate ;//l_singularity;//
    Triangulation<dim>   triangulation;
    const MappingQ1<dim> mapping;

    AffineConstraints<double> constraints;
    using ScratchData = MeshWorker::ScratchData<dim>;

    // Furthermore we want to use DG elements of degree 1 (but this is only
    // specified in the constructor). If you want to use a DG method of a
    // different degree the whole program stays the same, only replace 1 in
    // the constructor by the desired polynomial degree.
    FE_DGQ<dim>          fe;
    DoFHandler<dim>      dof_handler;

    // The next four members represent the linear system to be
    // solved. <code>system_matrix</code> and <code>right_hand_side</code> are
    // generated by <code>assemble_system()</code>, the <code>solution</code>
    // is computed in <code>solve()</code>. The <code>sparsity_pattern</code>
    // is used to determine the location of nonzero elements in
    // <code>system_matrix</code>.
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double>       solution;
    Vector<double>       system_rhs;
    Vector<double>       estimated_error_per_cell;
    ConvergenceTable convergence_table;
  };


  // We start with the constructor. The 1 in the constructor call of
  // <code>fe</code> is the polynomial degree.
  template <int dim>
  SIPGLaplace<dim>::SIPGLaplace ()
    :
    mapping (),
    fe (3),
    dof_handler (triangulation)
  {}


  template <int dim>
  void SIPGLaplace<dim>::setup_system ()
  {
    // In the function that sets up the usual finite element data structures,
    // we first need to distribute the DoFs.
    dof_handler.distribute_dofs (fe);

    // We start by generating the sparsity pattern. To this end, we first fill
    // an intermediate object of type DynamicSparsityPattern with the
    // couplings appearing in the system. After building the pattern, this
    // object is copied to <code>sparsity_pattern</code> and can be discarded.

    // To build the sparsity pattern for DG discretizations, we can call the
    // function analogue to DoFTools::make_sparsity_pattern, which is called
    // DoFTools::make_flux_sparsity_pattern:
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_flux_sparsity_pattern (dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    // Finally, we set up the structure of all components of the linear
    // system.
    system_matrix.reinit (sparsity_pattern); 
    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
    constraints.close();
  }

  template <int dim>
  double SIPGLaplace<dim>::compute_penalty (const double h1, const double h2)
  {
    const double degree = std::max(1.0, static_cast<double> (fe.get_degree()));
    return 4.0*degree*(degree+1.0)*0.5*(1.0/h1+1.0/h2);
  }



  template <int dim>
  void SIPGLaplace<dim>::assemble_system ()
  {

    typedef decltype(dof_handler.begin_active()) Iterator;
    const RHS<dim> rhs_function(test_case);
    const Viscosity<dim> viscosity_function;
    const Solution<dim> boundary_function(test_case);

    auto cell_worker = [&] (const Iterator &cell, 
                            ScratchData             &scratch_data, 
                            CopyData       &copy_data)
    {
      const FEValues<dim> &fe_v = scratch_data.reinit(cell); 
      const unsigned int dofs_per_cell   = fe_v.dofs_per_cell;
      copy_data.reinit(cell, dofs_per_cell);
      
      const auto &q_points = scratch_data.get_quadrature_points();
      const unsigned int n_q_points      = q_points.size();
      const std::vector<double> &JxW = scratch_data.get_JxW_values ();
      
      std::vector<double> nu (n_q_points);
      viscosity_function.value_list (q_points, nu);
      std::vector<double> rhs (n_q_points);
      rhs_function.value_list (q_points, rhs);

      for (unsigned int point=0; point<n_q_points; ++point)
        for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
          {
            for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
              copy_data.cell_matrix(i,j) +=
                  // nu \nabla u \nabla v
                  nu[point]
                  * fe_v.shape_grad(i,point)
                  * fe_v.shape_grad(j,point)
                  * JxW[point];

            copy_data.cell_rhs(i) += rhs[point] * fe_v.shape_value(i,point) * JxW[point];
          }
    };

    auto boundary_worker = [&] (const Iterator &cell, 
                                const unsigned int &face_no, 
                                ScratchData &scratch_data, 
                                CopyData &copy_data)
    {
      const FEFaceValuesBase<dim> &fe_fv =scratch_data.reinit(cell, face_no);
            
      const auto &q_points = scratch_data.get_quadrature_points();
      const unsigned int n_q_points = q_points.size();
      const unsigned int dofs_per_cell = fe_fv.dofs_per_cell;

      const std::vector<double> &JxW = scratch_data.get_JxW_values ();
      const std::vector<Tensor<1,dim> > &normals = scratch_data.get_normal_vectors ();

      std::vector<double> nu (n_q_points);
      viscosity_function.value_list (q_points, nu);

      std::vector<double> g(n_q_points);
      boundary_function.value_list (q_points, g);

      
      const double extent1 = cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[face_no]);
      const double penalty = compute_penalty(extent1, extent1);

      for (unsigned int point=0; point<n_q_points; ++point)
        {
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              copy_data.cell_matrix(i,j) +=
                  (
                    // - nu (\nabla u . n) v
                    - nu[point]
                    * (fe_fv.shape_grad(j,point) * normals[point])
                    * fe_fv.shape_value(i,point)

                    // - nu u (\nabla v . n)  // NIPG: use +
                    - nu[point]
                    * fe_fv.shape_value(j,point)
                    * (fe_fv.shape_grad(i,point) * normals[point])

                    // + nu * penalty u v
                    + nu[point]
                    * penalty
                    * fe_fv.shape_value(j,point)
                    * fe_fv.shape_value(i,point)
                    ) * JxW[point];

          for (unsigned int i=0; i<dofs_per_cell; ++i)
            copy_data.cell_rhs(i) +=
                (
                  // -nu g (\nabla v . n) // NIPG: use +
                  - nu[point]
                  * g[point]
                  * (fe_fv.shape_grad(i,point) * normals[point])

                  // +nu penalty g v
                  + nu[point]
                  * penalty
                  * g[point]
                  * fe_fv.shape_value(i,point)
                  ) * JxW[point];

        }
    };

    auto face_worker = [&]
                       (const Iterator &cell, const unsigned int &f, const unsigned int &sf,
                        const Iterator &ncell, const unsigned int &nf, const unsigned int &nsf,
                        ScratchData &scratch_data, CopyData &copy_data)
    {
      const FEInterfaceValues<dim> &fe_iv = scratch_data.reinit(cell,f,sf,ncell,nf,nsf);
      
      const auto &q_points = fe_iv.get_quadrature_points();
      const unsigned int n_q_points = q_points.size();

      copy_data.face_data.emplace_back();
      CopyDataFace &copy_data_face = copy_data.face_data.back();
      const unsigned int n_dofs_face        = fe_iv.n_current_interface_dofs();
      copy_data_face.joint_dof_indices = fe_iv.get_interface_dof_indices();
      copy_data_face.cell_matrix.reinit(n_dofs_face, n_dofs_face);

      const std::vector<double> &JxW = fe_iv.get_JxW_values ();
      const std::vector<Tensor<1,dim> > &normals = fe_iv.get_normal_vectors ();

      std::vector<double> nu (n_q_points);
      viscosity_function.value_list (q_points, nu);

      const double extent1 = cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[f])
          * (cell->has_children() ? 2.0 : 1.0);
      const double extent2 = ncell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[nf])
          * (ncell->has_children() ? 2.0 : 1.0);
      const double penalty =compute_penalty(extent1,extent2);

      for (unsigned int point=0; point<n_q_points; ++point)
        {
          for (unsigned int i=0; i<n_dofs_face; ++i)
            for (unsigned int j=0; j<n_dofs_face; ++j)
              copy_data_face.cell_matrix(i,j) +=
                  (
                  // - nu {\nabla u}.n [v] (consistency)
                  - nu[point]
                  * (fe_iv.average_gradient(j, point) * normals[point])
                  * fe_iv.jump(i, point)

                    // - nu [u] {\nabla v}.n  (symmetry) // NIPG: use +
                    - nu[point]
                    * fe_iv.jump(j, point)
                    * (fe_iv.average_gradient(i, point) * normals[point])

                    // nu sigma [u] [v] (penalty)
                    + nu[point] * penalty
                    * fe_iv.jump(j, point)
                    * fe_iv.jump(i, point)

                  ) * JxW[point];
        }

    };

    auto copier = [&] (const CopyData &c)
    {
      copy(c, constraints, system_matrix, system_rhs);
    };  

    const unsigned int n_gauss_points = dof_handler.get_fe().degree+1;
    QGauss<dim>     quadrature(n_gauss_points);
    QGauss<dim - 1> face_quadrature(n_gauss_points);

    UpdateFlags cell_flags = update_values | update_gradients |
                           update_quadrature_points | update_JxW_values;
    UpdateFlags face_flags = update_values | update_gradients |
                           update_quadrature_points | update_normal_vectors |
                           update_JxW_values;

    ScratchData scratch_data(mapping, fe, quadrature, cell_flags,
                            face_quadrature, face_flags);
    CopyData cd;
    MeshWorker::mesh_loop(dof_handler.begin_active(),
                          dof_handler.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          cd,
                          MeshWorker::assemble_own_cells |
                          MeshWorker::assemble_boundary_faces |
                          MeshWorker::assemble_own_interior_faces_once,
                          boundary_worker,
                          face_worker);


  }
  // @sect4{The assemble_system function}



  // Here we see the major difference to assembling by hand. Instead of
  // writing loops over cells and faces, we leave all this to the MeshWorker
  // framework. In order to do so, we just have to define local integration
  // functions and use one of the classes in namespace MeshWorker::Assembler
  // to build the global system.
  


  // @sect3{All the rest}
  //
  // For this simple problem we use the simplest possible solver, called
  // Richardson iteration, that represents a simple defect correction. This,
  // in combination with a block SSOR preconditioner, that uses the special
  // block matrix structure of system matrices arising from DG
  // discretizations. The size of these blocks are the number of DoFs per
  // cell. Here, we use a SSOR preconditioning as we have not renumbered the
  // DoFs according to the flow field. If the DoFs are renumbered in the
  // downstream direction of the flow, then a block Gauss-Seidel
  // preconditioner (see the PreconditionBlockSOR class with relaxation=1)
  // does a much better job.
  template <int dim>
  void SIPGLaplace<dim>::solve (Vector<double> &solution)
  {
    bool direct_solve = true;
    if(direct_solve)
      {
        std::cout << "   Solving system..." << std::endl;
        SparseDirectUMFPACK A_direct;
        A_direct.initialize(system_matrix);
        A_direct.vmult(solution, system_rhs);
      }
    else
      {
        SolverControl           solver_control (1000, 1e-12);
        SolverGMRES<>      solver (solver_control);

        // Here we create the preconditioner,
        // then assign the matrix to it and set the right block size:
        PreconditionBlockSSOR<SparseMatrix<double> > preconditioner;
        preconditioner.initialize(system_matrix, fe.dofs_per_cell);


        //PreconditionSSOR<SparseMatrix<double> > preconditioner;
        //preconditioner.initialize(system_matrix, fe.dofs_per_cell);

        // After these preparations we are ready to start the linear solver.
        solver.solve (system_matrix, solution, system_rhs,
                      preconditioner);
      }            
  }


  // We refine the grid according to a very simple refinement criterion,
  // namely an approximation to the gradient of the solution. As here we
  // consider the DG(1) method (i.e. we use piecewise bilinear shape
  // functions) we could simply compute the gradients on each cell. But we do
  // not want to base our refinement indicator on the gradients on each cell
  // only, but want to base them also on jumps of the discontinuous solution
  // function over faces between neighboring cells. The simplest way of doing
  // that is to compute approximative gradients by difference quotients
  // including the cell under consideration and its neighbors. This is done by
  // the <code>DerivativeApproximation</code> class that computes the
  // approximate gradients in a way similar to the
  // <code>GradientEstimation</code> described in step-9 of this tutorial. In
  // fact, the <code>DerivativeApproximation</code> class was developed
  // following the <code>GradientEstimation</code> class of step-9. Relating
  // to the discussion in step-9, here we consider $h^{1+d/2}|\nabla_h
  // u_h|$. Furthermore we note that we do not consider approximate second
  // derivatives because solutions to the linear advection equation are in
  // general not in $H^2$ but in $H^1$ (to be more precise, in $H^1_\beta$)
  // only.
  

  // The output of this program consists of eps-files of the adaptively
  // refined grids and the numerical solutions given in gnuplot format. This
  // was covered in previous examples and will not be further commented on.
  template <int dim>
  void SIPGLaplace<dim>::output_results (const unsigned int cycle) const
  {
    // Output of the solution in gnuplot format.
    std::string filename = "sol_Q" + std::to_string(fe.get_degree()) + "-";
    filename += ('0' + cycle );
    
    Assert (cycle < 10, ExcInternalError());

    filename += ".vtu";
    deallog << "Writing solution to <" << filename << ">" << std::endl;
    std::ofstream gnuplot_output (filename.c_str());

    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, "u", 
                              DataOut<dim>::type_dof_data);

    data_out.build_patches ();

    data_out.write_vtu(gnuplot_output);
  }

template <int dim>
void SIPGLaplace<dim>::compute_errors () 
  {
    double L2_error, H1_error;

    {
      Vector<float> difference_per_cell (triangulation.n_active_cells());
      VectorTools::integrate_difference (mapping,
                                         dof_handler,
                                         solution,
                                         Solution<dim>(test_case),
                                         difference_per_cell,
                                         QGauss<dim>(fe.degree+2),
                                         VectorTools::L2_norm);

      L2_error = VectorTools::compute_global_error(triangulation,
                                                   difference_per_cell,
                                                   VectorTools::L2_norm);
      std::cout << "   Error in the L2 norm       :     " << L2_error << std::endl;
    }

    {
      Vector<float> difference_per_cell (triangulation.n_active_cells());
      VectorTools::integrate_difference (mapping,
                                         dof_handler,
                                         solution,
                                         Solution<dim>(test_case),
                                         difference_per_cell,
                                         QGauss<dim>(fe.degree+2),
                                         VectorTools::H1_seminorm);

      H1_error = VectorTools::compute_global_error(triangulation,
                                                   difference_per_cell,
                                                   VectorTools::H1_seminorm);
      std::cout << "   Error in the H1 norm       :     " << H1_error << std::endl;
    }    

    convergence_table.add_value("L2", L2_error);
    convergence_table.add_value("H1", H1_error);
  }
  template <int dim>
  void SIPGLaplace<dim>::compute_error_estimate()
  {
    typedef decltype(dof_handler.begin_active()) Iterator;
    const RHS<dim> rhs_function(test_case);
    const Viscosity<dim> viscosity_function;
    const Solution<dim> boundary_function(test_case);
    estimated_error_per_cell.reinit(triangulation.n_active_cells());

    auto cell_worker = [&] (const Iterator &cell, 
                            ScratchData &scratch_data, 
                            CopyData &copy_data)
    {
      const FEValues<dim> &fe_v = scratch_data.reinit(cell);
            
      copy_data.cell_index = cell->active_cell_index();
      
      const auto & q_points = fe_v.get_quadrature_points();
      const unsigned int n_q_points = q_points.size();
      const std::vector<double> &JxW = fe_v.get_JxW_values ();

      std::vector<Tensor<2, dim>> hessians(n_q_points);
      fe_v.get_function_hessians(solution, hessians);

      std::vector<double> nu (n_q_points);
      viscosity_function.value_list (q_points, nu);
      std::vector<double> rhs (n_q_points);
      rhs_function.value_list (q_points, rhs);

      double residual_norm_square = 0;
      for (unsigned int point=0; point<n_q_points; ++point)
        {
          const double residual = rhs[point] + nu[point]*trace(hessians[point]);
          residual_norm_square += residual*residual*JxW[point];
        }
        copy_data.value =cell->diameter()*std::sqrt(  residual_norm_square);
    };

    auto boundary_worker = [&] (const Iterator &cell, 
                                const unsigned int &face_no, 
                                ScratchData &scratch_data, 
                                CopyData &copy_data)
    {
      const FEFaceValuesBase<dim> &fe_fv = scratch_data.reinit(cell,face_no);

      const auto &q_points = fe_fv.get_quadrature_points();
      const unsigned n_q_points = q_points.size();
      
      const std::vector<double> &JxW = fe_fv.get_JxW_values ();
      
      std::vector<double> nu (n_q_points);
      viscosity_function.value_list (q_points, nu);

      std::vector<double> g(n_q_points);
      boundary_function.value_list (q_points, g);

      std::vector<double> sol_u(n_q_points);
      fe_fv.get_function_values(solution, sol_u);

      const double degree = std::max(1.0, static_cast<double> (fe_fv.get_fe().degree));
      const double extent1 = cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[face_no]);
      const double penalty = compute_penalty(extent1,extent1);
     
      double difference_norm_square = 0;
      for (unsigned int point=0; point<q_points.size(); ++point)
        {
          const double diff= (g[point]-sol_u[point]);
          difference_norm_square += diff*diff*JxW[point];
        }
        copy_data.value +=std::sqrt(penalty*difference_norm_square);
    };

    auto face_worker = [&] (const Iterator &cell, 
                            const unsigned int &f, 
                            const unsigned int &sf,
                            const Iterator &ncell, 
                            const unsigned int &nf,
                            const unsigned int &nsf,
                            ScratchData &scratch_data,
                            CopyData &copy_data)
    {
      const FEInterfaceValues<dim> &fe_iv = scratch_data.reinit(cell, f, sf, ncell, nf, nsf);  
      
      copy_data.face_data.emplace_back();
      CopyDataFace &copy_data_face = copy_data.face_data.back();
      
      copy_data_face.cell_indices[0] = cell->active_cell_index();
      copy_data_face.cell_indices[1] = ncell->active_cell_index();

      const std::vector<double> &JxW = fe_iv.get_JxW_values ();
      const std::vector<Tensor<1,dim> > &normals = fe_iv.get_normal_vectors ();
      
      const auto &q_points = fe_iv.get_quadrature_points();
      const unsigned int n_q_points = q_points.size();

      std::vector<double> nu (n_q_points);
      viscosity_function.value_list (q_points, nu);
      
      std::vector<Tensor<1, dim>> grad_u[2];
      std::vector<double> sol_u[2];

      const double h = cell->face(f)->measure();
      for(unsigned int i = 0; i < 2; ++i)
      {
        grad_u[i].resize(n_q_points);
        sol_u[i].resize(n_q_points);
        fe_iv.get_fe_face_values(i).get_function_values(
        solution, sol_u[i]);
        fe_iv.get_fe_face_values(i).get_function_gradients(
        solution, grad_u[i]);
      }

      const double degree = std::max(1.0, static_cast<double>(fe_iv.get_fe().degree));
      const double extent1 = cell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[f])
          * (cell->has_children() ? 2.0 : 1.0);
      const double extent2 = ncell->extent_in_direction(GeometryInfo<dim>::unit_normal_direction[nf])
          * (ncell->has_children() ? 2.0 : 1.0);
      const double penalty = compute_penalty(extent1,extent2);

      double flux_jump_square=0;
      double u_jump_square =0;
      for (unsigned int point=0; point<n_q_points; ++point)
        {
          const double u_jump = sol_u[0][point]-sol_u[1][point];
          u_jump_square += u_jump*u_jump*JxW[point];

          const double flux_jump = (grad_u[0][point]-grad_u[1][point])*normals[point];
          flux_jump_square += nu[point]*flux_jump*flux_jump*JxW[point];
        }
        copy_data_face.values[0] = 0.5*std::sqrt(h*flux_jump_square + penalty*u_jump_square);
        copy_data_face.values[1] = copy_data_face.values[0];
    };
    
    auto copier = [&](const CopyData &copy_data) 
    {
      if (copy_data.cell_index != numbers::invalid_unsigned_int)
        estimated_error_per_cell[copy_data.cell_index] += copy_data.value;
      for (auto &cdf : copy_data.face_data)
        for (unsigned int j = 0; j < 2; ++j)
          estimated_error_per_cell[cdf.cell_indices[j]] += cdf.values[j];
    };

    const unsigned int n_gauss_points = dof_handler.get_fe().degree+1;
    QGauss<dim>     quadrature(n_gauss_points);
    QGauss<dim - 1> face_quadrature(n_gauss_points);

    UpdateFlags cell_flags = update_hessians | 
                             update_quadrature_points |
                             update_JxW_values;
    UpdateFlags face_flags = update_values |
                             update_gradients | 
                             update_quadrature_points |
                             update_JxW_values | 
                             update_normal_vectors;

    ScratchData scratch_data(mapping, fe, quadrature, cell_flags,
                            face_quadrature, face_flags);

    CopyData cd;
    MeshWorker::mesh_loop(dof_handler.begin_active(),
                          dof_handler.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          cd,
                          MeshWorker::assemble_own_cells |
                          MeshWorker::assemble_own_interior_faces_once|
                          MeshWorker::assemble_boundary_faces,
                          boundary_worker,
                          face_worker
                          );
    
  }




template <int dim>
  void SIPGLaplace<dim>::refine_grid ()
  {
    const double refinement_fraction = 0.2;

    GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                     estimated_error_per_cell,
                                                     refinement_fraction, 0.);

    triangulation.execute_coarsening_and_refinement ();
  }

 
  template <int dim>
  void SIPGLaplace<dim>::run ()
  {
    unsigned int max_cycle = test_case==convergence_rate? 6: 9;
    for (unsigned int cycle=0; cycle<max_cycle; ++cycle)
      {
        deallog << "Cycle " << cycle << std::endl;

        switch (test_case)
          {
            case convergence_rate:
            {
              if (cycle == 0)
                {
                  GridGenerator::hyper_cube (triangulation);
                  
                  triangulation.refine_global (2);
                }
              else{
                  triangulation.refine_global (1);
                  }              
              break;
            }
            case l_singularity:
            {
              if (cycle == 0)
                {
                  GridGenerator::hyper_L (triangulation);
                  triangulation.refine_global (1);
                }
              else
                {
                  refine_grid ();
                }

              
            }
          }

          


        deallog << "Number of active cells:       "
                << triangulation.n_active_cells()
                << std::endl;

        setup_system ();

        deallog << "Number of degrees of freedom: "
                << dof_handler.n_dofs()
                << std::endl;

        assemble_system ();
        //assemble_system ();
        solve (solution);
        compute_error_estimate();

        output_results (cycle);
        {
          convergence_table.add_value("cycle", cycle);
          convergence_table.add_value("cells", triangulation.n_active_cells());
          convergence_table.add_value("dofs", dof_handler.n_dofs());
        }
        compute_errors();

        if(test_case == l_singularity)
          convergence_table.add_value("Estimator", estimated_error_per_cell.l2_norm());
        
        std::cout<<std::endl;
      }
      {
        convergence_table.set_precision("L2", 3);
        convergence_table.set_precision("H1", 3);
        
        convergence_table.set_scientific("L2", true);
        convergence_table.set_scientific("H1", true);
        
        if(test_case == l_singularity)
        {
          convergence_table.set_precision("Estimator", 3);
          convergence_table.set_scientific("Estimator", true);
          convergence_table.set_tex_caption("Estimator", "Estimated error");
        }
        if(test_case == convergence_rate)
        {
          convergence_table.evaluate_convergence_rates(
            "L2", ConvergenceTable::reduction_rate_log2);
          convergence_table.evaluate_convergence_rates(
            "H1", ConvergenceTable::reduction_rate_log2);

        }

        std::cout<< "degree = "<<fe.get_degree() << std::endl;
        convergence_table.write_text(std::cout,
                                    TableHandler::TextOutputFormat::org_mode_table);
      }
  }
}


// The following <code>main</code> function is similar to previous examples as
// well, and need not be commented on.
int main ()
{
  try
    {
      Step74::SIPGLaplace<2> problem;
      problem.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    };

  return 0;
}
