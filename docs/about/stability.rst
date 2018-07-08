.. _subpackage-stability:

************************
Stability of Subpackages
************************

.. This document is derived from docs/stability.rst in Astropy.  See
   licenses/Astropy_LICENSE.rst in PlasmaPy's directory.

This document summarizes the stability of PlasmaPy's subpackages so that
users understand where they might expect changes in the future, and
which subpackages they can safely use for production code.  Please note
that backward compatibility is not guaranteed for the `0.*.*` series of
development releases.  Starting with version `1.0.0`, the major version
number will be incremented when a release contains backward incompatible
changes.

The classification is as follows:

.. raw:: html

    <style>
         .planned:before {
              color: #cbcbcb;
              content: "⬤";
         }
         .dev:before {
              color: #ffad00;
              content: "⬤";
         }
         .stable:before {
              color: #4e72c3;
              content: "⬤";
         }
         .mature:before {
              color: #03a913;
              content: "⬤";
         }
         .pendingdep:before {
              color: #a84b03;
              content: "⬤";
         }
         .deprecated:before {
              color: #ff0000;
              content: "⬤";
         }
    </style>

    <table align='center'>
      <tr>
        <td align='center'><span class="planned"></span></td>
        <td>Planned</td>
      </tr>
      <tr>
        <td align='center'><span class="dev"></span></td>
        <td>Actively being developed. Be prepared for possible significant changes.</td>
      </tr>
      <tr>
        <td align='center'><span class="stable"></span></td>
        <td>Reasonably stable. Any significant changes/additions will generally include backwards-compatiblity.</td>
      </tr>
      <tr>
        <td align='center'><span class="mature"></span></td>
        <td>Mature.  Additions/improvements possible, but no major changes planned. </td>
      </tr>
      <tr>
        <td align='center'><span class="pendingdep"></span></td>
        <td>Pending deprecation.  Might be deprecated in a future version.</td>
      </tr>
      <tr>
        <td align='center'><span class="deprecated"></span></td>
        <td>Deprecated.  Might be removed in a future version.</td>
      </tr>
    </table>

PlasmaPy's planned and existing subpackages are:

.. raw:: html

    <table border="1" class="docutils stability" align='center'>
        <tr>
            <th class="head">
                Subpackage
            </th>
            <th class="head">
                &nbsp;
            </th>
            <th class="head">
                Comments
            </th>
        </tr>
        <tr>
            <td>
                plasmapy.atomic
            </td>
            <td align='center'>
                <span class="dev"></span>
            </td>
            <td>
                This package is being actively developed and expanded,
                and there may be backward incompatible changes to the API.
            </td>
        </tr>
        <tr>
            <td>
                plasmapy.classes
            </td>
            <td align='center'>
                <span class="dev"></span>
            </td>
            <td>
                The plan for PlasmaPy's base classes is being planned in
                PLEP 7, which is in the process of being written.  The
                existing functionality is unstable.
            </td>
        </tr>
        <tr>
            <td>
                plasmapy.constants
            </td>
            <td align='center'>
                <span class="stable"></span>
            </td>
            <td>
                We do not anticipate that there will be any major backward
                incompatible changes within the
                <tt class="docutils literal"><span class="pre">constants</span></tt>
                subpackage.  However, the values for constants may be updated
                when improved values become available.
            </td>
        </tr>
        <tr>
            <td>
                plasmapy.diagnostics
            </td>
            <td align='center'>
                <span class="dev"></span>
            </td>
            <td>
                This subpackage is in the early stages of development.
            </td>
        </tr>
        <tr>
            <td>
                plasmapy.mathematics
            </td>
            <td align='center'>
                <span class="dev"></span>
            </td>
            <td>
                The
                <tt class="docutils literal"><span class="pre">mathematics</span></tt>
                subpackage is in the early stages of development.
            </td>
        </tr>
        <tr>
            <td>
                plasmapy.physics
            </td>
            <td align='center'>
                <span class="dev"></span>
            </td>
            <td>
                The
                <tt class="docutils literal"><span class="pre">physics</span></tt>
                subpackage may undergo significant reorganization with potentially
                major changes to the API.  We anticipate that there will be major
                additions to this package.
            </td>
        </tr>
        <tr>
            <td>
                plasmapy.physics.transport
            </td>
            <td align='center'>
                <span class="dev"></span>
            </td>
            <td>
                The
                <tt class="docutils literal"><span class="pre">transport</span></tt>
                subpackage will likely undergo significant reorganization
                and expansion with major changes to the API.
            </td>
        </tr>
        <tr>
            <td>
                plasmapy.utils
            </td>
            <td align='center'>
                <span class="dev"></span>
            </td>
            <td>
                The
                <tt class="docutils literal"><span class="pre">utils</span></tt>
                subpackage may undergo significant reorganization with potentially
                major changes to the API.
            </td>
        </tr>
    </table>
