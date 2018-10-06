// ---------------------------------------------------------------------
//
// Copyright (C) 1998 - 2018 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------

#include <deal.II/base/tensor.h>
#include <deal.II/base/std_cxx14/memory.h>
#include <deal.II/grid/tria_boundary.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <cmath>

DEAL_II_NAMESPACE_OPEN



/* -------------------------- Boundary --------------------- */


template <int dim, int spacedim>
std::unique_ptr<Manifold<dim, spacedim> >
Boundary<dim,spacedim>::clone() const
{
  AssertThrow(false, ExcPureFunctionCalled());
  return std::unique_ptr<Manifold<dim,spacedim> >();
}



template <int dim, int spacedim>
void
Boundary<dim, spacedim>::
get_intermediate_points_on_line (const typename Triangulation<dim, spacedim>::line_iterator &,
                                 std::vector<Point<spacedim> > &) const
{
  Assert (false, ExcPureFunctionCalled());
}



template <int dim, int spacedim>
void
Boundary<dim, spacedim>::
get_intermediate_points_on_quad (const typename Triangulation<dim, spacedim>::quad_iterator &,
                                 std::vector<Point<spacedim> > &) const
{
  Assert (false, ExcPureFunctionCalled());
}


template <int dim, int spacedim>
void
Boundary<dim,spacedim>::
get_intermediate_points_on_face (const typename Triangulation<dim,spacedim>::face_iterator &face,
                                 std::vector<Point<spacedim> > &points) const
{
  Assert (dim>1, ExcImpossibleInDim(dim));

  switch (dim)
    {
    case 2:
      get_intermediate_points_on_line (face, points);
      break;
    case 3:
      get_intermediate_points_on_quad (face, points);
      break;
    default:
      Assert (false, ExcNotImplemented());
    }
}


template <>
void
Boundary<1,1>::
get_intermediate_points_on_face (const Triangulation<1,1>::face_iterator &,
                                 std::vector<Point<1> > &) const
{
  Assert (false, ExcImpossibleInDim(1));
}


template <>
void
Boundary<1,2>::
get_intermediate_points_on_face (const Triangulation<1,2>::face_iterator &,
                                 std::vector<Point<2> > &) const
{
  Assert (false, ExcImpossibleInDim(1));
}


template <>
void
Boundary<1,3>::
get_intermediate_points_on_face (const Triangulation<1,3>::face_iterator &,
                                 std::vector<Point<3> > &) const
{
  Assert (false, ExcImpossibleInDim(1));
}


template <int dim, int spacedim>
Point<spacedim>
Boundary<dim, spacedim>::
project_to_surface (const typename Triangulation<dim, spacedim>::line_iterator &,
                    const Point<spacedim>                                &trial_point) const
{
  if (spacedim <= 1)
    return trial_point;
  else
    {
      Assert (false, ExcPureFunctionCalled());
      return Point<spacedim>();
    }
}



template <int dim, int spacedim>
Point<spacedim>
Boundary<dim, spacedim>::
project_to_surface (const typename Triangulation<dim, spacedim>::quad_iterator &,
                    const Point<spacedim>                                &trial_point) const
{
  if (spacedim <= 2)
    return trial_point;
  else
    {
      Assert (false, ExcPureFunctionCalled());
      return Point<spacedim>();
    }
}



template <int dim, int spacedim>
Point<spacedim>
Boundary<dim, spacedim>::
project_to_surface (const typename Triangulation<dim, spacedim>::hex_iterator &,
                    const Point<spacedim>                                &trial_point) const
{
  if (spacedim <= 3)
    return trial_point;
  else
    {
      Assert (false, ExcPureFunctionCalled());
      return Point<spacedim>();
    }
}



template <int dim, int spacedim>
const std::vector<Point<1> > &
Boundary<dim,spacedim>::
get_line_support_points (const unsigned int n_intermediate_points) const
{
  if (points.size() <= n_intermediate_points ||
      points[n_intermediate_points].get() == nullptr)
    {
      Threads::Mutex::ScopedLock lock(mutex);
      if (points.size() <= n_intermediate_points)
        points.resize(n_intermediate_points+1);

      // another thread might have created points in the meantime
      if (points[n_intermediate_points].get() == nullptr)
        points[n_intermediate_points] = std_cxx14::make_unique<QGaussLobatto<1> >
                                        (n_intermediate_points+2);
    }
  return points[n_intermediate_points]->get_points();
}




/* -------------------------- StraightBoundary --------------------- */

// At least clang < 3.9.0 complains if we move this definition to its
// declaration when a 'const StraightBoundary' object is built.
template <int dim, int spacedim>
StraightBoundary<dim, spacedim>::StraightBoundary () = default;



template <int dim, int spacedim>
Point<spacedim>
StraightBoundary<dim, spacedim>::
get_new_point_on_line (const typename Triangulation<dim, spacedim>::line_iterator &line) const
{
  return (line->vertex(0) + line->vertex(1)) / 2;
}


template <int dim, int spacedim>
std::unique_ptr<Manifold<dim, spacedim> >
StraightBoundary<dim,spacedim>::clone() const
{
  return std_cxx14::make_unique<StraightBoundary<dim,spacedim> >();
}


namespace
{
  // compute the new midpoint of a quad --
  // either of a 2d cell on a manifold in 3d
  // or of a face of a 3d triangulation in 3d
  template <int dim>
  Point<3>
  compute_new_point_on_quad (const typename Triangulation<dim, 3>::quad_iterator &quad)
  {
    // generate a new point in the middle of
    // the face based on the points on the
    // edges and the vertices.
    //
    // there is a pathological situation when
    // this face is on a straight boundary, but
    // one of its edges and the face behind it
    // are not; if that face is refined first,
    // the new point in the middle of that edge
    // may not be at the same position as
    // quad->line(.)->center() would have been,
    // but would have been moved to the
    // non-straight boundary. We cater to that
    // situation by using existing edge
    // midpoints if available, or center() if
    // not
    //
    // note that this situation can not happen
    // during mesh refinement, as there the
    // edges are refined first and only then
    // the face. thus, the check whether a line
    // has children does not lead to the
    // situation where the new face midpoints
    // have different positions depending on
    // which of the two cells is refined first.
    //
    // the situation where the edges aren't
    // refined happens when a higher order
    // MappingQ requests the midpoint of a
    // face, though, and it is for these cases
    // that we need to have the check available
    //
    // note that the factor of 1/8 for each
    // of the 8 surrounding points isn't
    // chosen arbitrarily. rather, we may ask
    // where the harmonic map would place the
    // point (0,0) if we map the square
    // [-1,1]^2 onto the domain that is
    // described using the 4 vertices and 4
    // edge point points of this quad. we can
    // then discretize the harmonic map using
    // four cells and Q1 elements on each of
    // the quadrants of the square [-1,1]^2
    // and see where the midpoint would land
    // (this is the procedure we choose, for
    // example, in
    // GridGenerator::laplace_solve) and it
    // turns out that it will land at the
    // mean of the 8 surrounding
    // points. whether a discretization of
    // the harmonic map with only 4 cells is
    // adequate is a different question
    // altogether, of course.
    return (quad->vertex(0) + quad->vertex(1) +
            quad->vertex(2) + quad->vertex(3) +
            (quad->line(0)->has_children() ?
             quad->line(0)->child(0)->vertex(1) :
             quad->line(0)->center()) +
            (quad->line(1)->has_children() ?
             quad->line(1)->child(0)->vertex(1) :
             quad->line(1)->center()) +
            (quad->line(2)->has_children() ?
             quad->line(2)->child(0)->vertex(1) :
             quad->line(2)->center()) +
            (quad->line(3)->has_children() ?
             quad->line(3)->child(0)->vertex(1) :
             quad->line(3)->center())               ) / 8;
  }
}



template <int dim, int spacedim>
Point<spacedim>
StraightBoundary<dim, spacedim>::
get_new_point_on_quad (const typename Triangulation<dim, spacedim>::quad_iterator &quad) const
{
  return FlatManifold<dim,spacedim>::get_new_point_on_quad(quad);
}


template <>
Point<3>
StraightBoundary<2,3>::
get_new_point_on_quad (const Triangulation<2,3>::quad_iterator &quad) const
{
  return compute_new_point_on_quad<2> (quad);
}



template <>
Point<3>
StraightBoundary<3>::
get_new_point_on_quad (const Triangulation<3>::quad_iterator &quad) const
{
  return compute_new_point_on_quad<3> (quad);
}



template <int dim, int spacedim>
void
StraightBoundary<dim, spacedim>::
get_intermediate_points_on_line (const typename Triangulation<dim, spacedim>::line_iterator &line,
                                 std::vector<Point<spacedim> > &points) const
{
  const unsigned int n=points.size();
  Assert(n>0, ExcInternalError());

  // Use interior points of QGaussLobatto quadrature formula support points
  // for consistency with MappingQ
  const std::vector<Point<1> > &line_points = this->get_line_support_points(n);

  const Point<spacedim> vertices[2] = { line->vertex(0),
                                        line->vertex(1)
                                      };

  for (unsigned int i=0; i<n; ++i)
    {
      const double x = line_points[1+i][0];
      points[i] = (1-x)*vertices[0] + x*vertices[1];
    }
}




template <int dim, int spacedim>
void
StraightBoundary<dim, spacedim>::
get_intermediate_points_on_quad (const typename Triangulation<dim, spacedim>::quad_iterator &,
                                 std::vector<Point<spacedim> > &) const
{
  Assert(false, ExcImpossibleInDim(dim));
}



template <>
void
StraightBoundary<3>::
get_intermediate_points_on_quad (const Triangulation<3>::quad_iterator &quad,
                                 std::vector<Point<3> > &points) const
{
  const unsigned int spacedim = 3;

  const unsigned int n=points.size(),
                     m=static_cast<unsigned int>(std::sqrt(static_cast<double>(n)));
  // is n a square number
  Assert(m*m==n, ExcInternalError());

  const std::vector<Point<1> > &line_points = this->get_line_support_points(m);

  const Point<spacedim> vertices[4] = { quad->vertex(0),
                                        quad->vertex(1),
                                        quad->vertex(2),
                                        quad->vertex(3)
                                      };

  for (unsigned int i=0; i<m; ++i)
    {
      const double y=line_points[1+i][0];
      for (unsigned int j=0; j<m; ++j)
        {
          const double x=line_points[1+j][0];
          points[i*m+j]=((1-x) * vertices[0] +
                         x     * vertices[1]) * (1-y) +
                        ((1-x) * vertices[2] +
                         x     * vertices[3]) * y;
        }
    }
}



template <>
void
StraightBoundary<2,3>::
get_intermediate_points_on_quad (const Triangulation<2,3>::quad_iterator &quad,
                                 std::vector<Point<3> > &points) const
{
  const unsigned int spacedim = 3;

  const unsigned int n=points.size(),
                     m=static_cast<unsigned int>(std::sqrt(static_cast<double>(n)));
  // is n a square number
  Assert(m*m==n, ExcInternalError());

  const std::vector<Point<1> > &line_points = this->get_line_support_points(m);

  const Point<spacedim> vertices[4] = { quad->vertex(0),
                                        quad->vertex(1),
                                        quad->vertex(2),
                                        quad->vertex(3)
                                      };

  for (unsigned int i=0; i<m; ++i)
    {
      const double y=line_points[1+i][0];
      for (unsigned int j=0; j<m; ++j)
        {
          const double x=line_points[1+j][0];
          points[i*m+j]=((1-x) * vertices[0] +
                         x     * vertices[1]) * (1-y) +
                        ((1-x) * vertices[2] +
                         x     * vertices[3]) * y;
        }
    }
}



template <int dim, int spacedim>
Tensor<1,spacedim>
StraightBoundary<dim,spacedim>::
normal_vector (const typename Triangulation<dim,spacedim>::face_iterator &face,
               const Point<spacedim>                                     &p) const
{
  return FlatManifold<dim, spacedim>::normal_vector(face, p);
}



template <int dim, int spacedim>
void
StraightBoundary<dim, spacedim>::
get_normals_at_vertices (const typename Triangulation<dim, spacedim>::face_iterator &face,
                         typename Boundary<dim,spacedim>::FaceVertexNormals         &face_vertex_normals) const
{
  FlatManifold<dim, spacedim>::get_normals_at_vertices(face, face_vertex_normals);
}



template <int dim, int spacedim>
Point<spacedim>
StraightBoundary<dim, spacedim>::
project_to_surface (const typename Triangulation<dim, spacedim>::line_iterator &line,
                    const Point<spacedim>                                &trial_point) const
{
  if (spacedim <= 1)
    return trial_point;
  else
    {
      // find the point that lies on
      // the line p1--p2. the
      // formulas pan out to
      // something rather simple
      // because the mapping to the
      // line is linear
      const Point<spacedim> p1 = line->vertex(0),
                            p2 = line->vertex(1);
      const double s = (trial_point-p1)*(p2-p1) / ((p2-p1)*(p2-p1));
      return p1 + s*(p2-p1);
    }
}



namespace internal
{
  template <typename Iterator, int spacedim, int dim>
  Point<spacedim>
  compute_projection (const Iterator        &object,
                      const Point<spacedim> &y,
                      std::integral_constant<int, dim>)
  {
    // let's look at this for
    // simplicity for a quad (dim==2)
    // in a space with spacedim>2:

    // all points on the surface are given by
    //   x(\xi) = sum_i v_i phi_x(\xi)
    // where v_i are the vertices of the quad,
    // and \xi=(\xi_1,\xi_2) are the reference
    // coordinates of the quad. so what we are
    // trying to do is find a point x on
    // the surface that is closest to the point
    // y. there are different ways
    // to solve this problem, but in the end
    // it's a nonlinear problem and we have to
    // find reference coordinates \xi so that
    //   J(\xi) = 1/2 || x(\xi)-y ||^2
    // is minimal. x(\xi) is a function that
    // is dim-linear in \xi, so J(\xi) is
    // a polynomial of degree 2*dim that
    // we'd like to minimize. unless dim==1,
    // we'll have to use a Newton
    // method to find the
    // answer. This leads to the
    // following formulation of
    // Newton steps:
    //
    // Given \xi_k, find \delta\xi_k so that
    //   H_k \delta\xi_k = - F_k
    // where H_k is an approximation to the
    // second derivatives of J at \xi_k, and
    // F_k is the first derivative of J.
    // We'll iterate this a number of times
    // until the right hand side is small
    // enough. As a stopping criterion, we
    // terminate if ||\delta\xi||<eps.
    //
    // As for the Hessian, the best choice
    // would be
    //   H_k = J''(\xi_k)
    // but we'll opt for the simpler
    // Gauss-Newton form
    //   H_k = A^T A
    // i.e.
    //   (H_k)_{nm} = \sum_{i,j} v_i*v_j *
    //                   \partial_n phi_i *
    //                   \partial_m phi_j
    // we start at xi=(0.5,0.5).
    Point<dim> xi;
    for (unsigned int d=0; d<dim; ++d)
      xi[d] = 0.5;

    Point<spacedim> x_k;
    for (unsigned int i=0; i<GeometryInfo<dim>::vertices_per_cell; ++i)
      x_k += object->vertex(i) *
             GeometryInfo<dim>::d_linear_shape_function (xi, i);

    do
      {
        Tensor<1,dim> F_k;
        for (unsigned int i=0; i<GeometryInfo<dim>::vertices_per_cell; ++i)
          F_k += (x_k-y)*object->vertex(i) *
                 GeometryInfo<dim>::d_linear_shape_function_gradient (xi, i);

        Tensor<2,dim> H_k;
        for (unsigned int i=0; i<GeometryInfo<dim>::vertices_per_cell; ++i)
          for (unsigned int j=0; j<GeometryInfo<dim>::vertices_per_cell; ++j)
            {
              Tensor<2, dim> tmp = outer_product(
                                     GeometryInfo<dim>::d_linear_shape_function_gradient(xi, i),
                                     GeometryInfo<dim>::d_linear_shape_function_gradient(xi, j));
              H_k += (object->vertex(i) * object->vertex(j)) * tmp;
            }

        const Tensor<1,dim> delta_xi = - invert(H_k) * F_k;
        xi += delta_xi;

        x_k = Point<spacedim>();
        for (unsigned int i=0; i<GeometryInfo<dim>::vertices_per_cell; ++i)
          x_k += object->vertex(i) *
                 GeometryInfo<dim>::d_linear_shape_function (xi, i);

        if (delta_xi.norm() < 1e-5)
          break;
      }
    while (true);

    return x_k;
  }


  // specialization for a quad in 1d
  template <typename Iterator>
  Point<1>
  compute_projection (const Iterator &,
                      const Point<1> &y,
                      /* it's a quad: */std::integral_constant<int, 2>)
  {
    return y;
  }

  // specialization for a quad in 2d
  template <typename Iterator>
  Point<2>
  compute_projection (const Iterator &,
                      const Point<2> &y,
                      /* it's a quad: */std::integral_constant<int, 2>)
  {
    return y;
  }
}





template <>
Point<3>
StraightBoundary<1,3>::
project_to_surface (const Triangulation<1, 3>::quad_iterator &,
                    const Point<3>  &y) const
{
  return y;
}

//TODO[SP]: This is just a horrible way out to make it compile in codim 2.
template <int dim, int spacedim>
Point<spacedim>
StraightBoundary<dim, spacedim>::
project_to_surface (const typename Triangulation<dim, spacedim>::quad_iterator &quad,
                    const Point<spacedim>  &y) const
{
  if (spacedim <= 2)
    return y;
  else
    return internal::compute_projection (quad, y,
                                         /* it's a quad */std::integral_constant<int, 2>());
}



template <int dim, int spacedim>
Point<spacedim>
StraightBoundary<dim, spacedim>::
project_to_surface (const typename Triangulation<dim, spacedim>::hex_iterator &,
                    const Point<spacedim>                                &trial_point) const
{
  if (spacedim <= 3)
    return trial_point;
  else
    {
      // we can presumably call the
      // same function as above (it's
      // written in a generic way)
      // but someone needs to check
      // whether that actually yields
      // the correct result
      Assert (false, ExcNotImplemented());
      return Point<spacedim>();
    }
}



// explicit instantiations
#include "tria_boundary.inst"

DEAL_II_NAMESPACE_CLOSE