/**
 * Combined capture-geometry viewer.
 *
 * One Three.js scene that holds:
 *   • Turntable disc + axis
 *   • Camera positions (blue scatter, optional arc, blue selection marker)
 *   • Light  positions (orange scatter, orange selection marker)
 *
 * The caller decides which coordinate frame the positions live in (base1
 * or world0) and where the turntable centre is in that frame — the scene
 * itself does not transform anything, it just draws what it's given.
 */

(() => {

    class CombinedCaptureScene {
        /**
         * @param {HTMLCanvasElement} canvas
         * @param {object} opts
         *   - cameraPositions : array of [x,y,z] (m)
         *   - lightPositions  : array of [x,y,z] (m)
         *   - turntableCenter : [x,y,z] (m) — disc centre
         *   - turntableRadius : disc radius (m), default 0.075
         *   - drawArc         : draw a polyline through cameraPositions
         *                       sorted by polar angle around the disc
         *   - camColor        : hex for camera scatter (default blue)
         *   - lightColor      : hex for light scatter  (default orange)
         *   - camSelectionColor / lightSelectionColor : selected markers
         */
        constructor(canvas, opts) {
            this.canvas = canvas;
            this.opts = opts;
            this.THREE = window.THREE;
            this._build();
        }

        _build() {
            const THREE = this.THREE;
            const {
                cameraPositions = [],
                lightPositions  = [],
                turntableCenter,
                turntableRadius    = 0.075,
                turntableThickness = 0.020,   // disc height, m
                drawArc           = true,
                camColor          = 0x3367d6,
                lightColor        = 0xf5b942,
                camSelectionColor = 0x0a3aad,
                lightSelectionColor = 0xffae00,
                // Optional material 3D bbox (world0 frame) — see bbox.json.
                materialBBox       = null,
                // If true (base1 view), the box rotates by +turn_angle
                // about the calibrated turntable axis/centre on each
                // setMaterialTurnAngle() call. If false, the box stays
                // axis-aligned at its world0 pose (used in world0 panel).
                rotateMaterialWithTurnAngle = false,
            } = this.opts;
            this._rotateMaterialBox = !!rotateMaterialWithTurnAngle;

            const renderer = new THREE.WebGLRenderer({
                canvas: this.canvas, antialias: true, alpha: true,
            });
            renderer.setPixelRatio(window.devicePixelRatio || 1);
            const w = this.canvas.clientWidth || 360;
            const h = this.canvas.clientHeight || 360;
            renderer.setSize(w, h, false);
            renderer.setClearColor(0xf6f7f9, 1);

            const scene = new THREE.Scene();
            this._center = turntableCenter;

            // ---- Frame the camera based on the bounding sphere of all points ----
            const allPts = [...cameraPositions, ...lightPositions, turntableCenter];
            const xs = allPts.map(p => p[0]);
            const ys = allPts.map(p => p[1]);
            const zs = allPts.map(p => p[2]);
            const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
            const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
            const cz = (Math.min(...zs) + Math.max(...zs)) / 2;
            const fitR = Math.max(
                0.05,
                ...allPts.map(p => Math.hypot(p[0]-cx, p[1]-cy, p[2]-cz)),
            );

            const camera = new THREE.PerspectiveCamera(35, w / h, 0.001, 50);
            camera.position.set(cx + fitR * 1.6, cy - fitR * 1.6, cz + fitR * 1.6);
            camera.up.set(0, 0, 1);
            camera.lookAt(new THREE.Vector3(cx, cy, cz));

            // ---- Turntable disc + rim + axis ----
            //
            // The caller passes turntableCenter with z = material bbox_min[2]
            // (the top surface of the disc — see obj.js). We render the
            // cylinder with its TOP face at that z, so the material slab
            // ends up sitting on it. Cylinder centre = top - thickness/2.
            const discTopZ = turntableCenter[2];
            const discCenterZ = discTopZ - turntableThickness / 2;
            const tt = new THREE.Mesh(
                new THREE.CylinderGeometry(
                    turntableRadius, turntableRadius, turntableThickness, 64,
                ),
                new THREE.MeshBasicMaterial({ color: 0xffffff }),
            );
            tt.rotation.x = Math.PI / 2;
            tt.position.set(turntableCenter[0], turntableCenter[1], discCenterZ);
            scene.add(tt);

            // Crisp rim along the top edge so the white disc reads against
            // the (slightly off-white) page background.
            const ttRim = new THREE.Mesh(
                new THREE.RingGeometry(turntableRadius * 0.99, turntableRadius, 64),
                new THREE.MeshBasicMaterial({ color: 0xb0b6be, side: THREE.DoubleSide }),
            );
            ttRim.position.set(turntableCenter[0], turntableCenter[1], discTopZ + 0.0005);
            scene.add(ttRim);

            // +Z axis indicator: short orange shaft + small cone arrow head.
            const axisLen   = turntableRadius * 0.42;   // total length (shaft + head)
            const headLen   = Math.max(0.008, axisLen * 0.18);
            const shaftLen  = axisLen - headLen;
            const shaftTopZ = discTopZ + shaftLen;
            // shaft
            scene.add(new THREE.Line(
                new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(turntableCenter[0], turntableCenter[1], discTopZ),
                    new THREE.Vector3(turntableCenter[0], turntableCenter[1], shaftTopZ),
                ]),
                new THREE.LineBasicMaterial({ color: 0xff7a45 }),
            ));
            // arrow head — ConeGeometry has apex at +Y, so rotateX(+π/2)
            // pivots +Y to +Z. The base then sits at shaftTopZ and the
            // apex points along +Z.
            const arrowGeom = new THREE.ConeGeometry(headLen * 0.35, headLen, 14);
            arrowGeom.rotateX(Math.PI / 2);
            const arrowHead = new THREE.Mesh(
                arrowGeom,
                new THREE.MeshBasicMaterial({ color: 0xff7a45 }),
            );
            arrowHead.position.set(
                turntableCenter[0], turntableCenter[1], shaftTopZ + headLen / 2,
            );
            scene.add(arrowHead);

            // (No decorative hemisphere wireframe: the THREE.SphereGeometry
            // default orientation has its pole along +Y but our turntable
            // axis is along +Z, so the wireframe's base ring landed in a
            // tilted plane and looked like a second, mis-aligned turntable.
            // The scatter itself is enough to convey the hemisphere shape.)

            // ---- Material 3D bounding box ----
            // bbox.json stores the cropped material's xyz min/max in WORLD0.
            // In the world0 view the box is drawn axis-aligned at its
            // stored pose. In the base1 view the box's initial pose is the
            // same (world0 pose), and we rotate it by +turn_angle around
            // the turntable axis whenever setMaterialTurnAngle() is called.
            if (materialBBox && materialBBox.bbox_min && materialBBox.bbox_max) {
                this._addMaterialBox(scene, materialBBox);
            }

            // ---- Camera-position scatter ----
            // Small THREE.Points dots — these convey trajectory coverage,
            // not individual cameras. The currently-selected camera gets a
            // larger 3D icon (see selCam below).
            if (cameraPositions.length > 0) {
                scene.add(new THREE.Points(
                    new THREE.BufferGeometry().setFromPoints(
                        cameraPositions.map(p => new THREE.Vector3(...p)),
                    ),
                    new THREE.PointsMaterial({
                        color: camColor, size: 0.010, sizeAttenuation: true,
                    }),
                ));
            }

            // ---- Light-position scatter ----
            if (lightPositions.length > 0) {
                scene.add(new THREE.Points(
                    new THREE.BufferGeometry().setFromPoints(
                        lightPositions.map(p => new THREE.Vector3(...p)),
                    ),
                    new THREE.PointsMaterial({
                        color: lightColor, size: 0.012, sizeAttenuation: true,
                    }),
                ));
            }

            // ---- Selected markers ----
            // These start as small sphere placeholders. obj.js asynchronously
            // loads two glTF models (AntiqueCamera + Lantern) and calls
            // attachSelectionModels() when ready, which swaps the spheres
            // out for the 3D models.
            this.selCam   = this._addMarker(scene, camSelectionColor,   0.013, turntableCenter);
            this.selLight = this._addMarker(scene, lightSelectionColor, 0.018, turntableCenter);

            // ---- Dashed selection rays back to the turntable centre ----
            this.camRay   = this._addRay(scene, camSelectionColor);
            this.lightRay = this._addRay(scene, lightSelectionColor);

            // ---- OrbitControls (optional) ----
            let controls = null;
            if (typeof THREE.OrbitControls === "function") {
                controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.target.set(cx, cy, cz);
                controls.update();
            }

            this.scene = scene;
            this.camera = camera;
            this.renderer = renderer;
            this.controls = controls;

            const animate = () => {
                requestAnimationFrame(animate);
                if (this.controls) this.controls.update();
                this.renderer.render(this.scene, this.camera);
            };
            animate();

            const ro = new ResizeObserver(() => this.resize());
            ro.observe(this.canvas);
        }

        /**
         * Camera-icon scatter: a small cone (the "pinhole") per camera,
         * each one rotated so its apex points at the turntable centre.
         * One InstancedMesh keeps draw calls flat.
         */
        _addCameraIcons(scene, positions, color, target) {
            const THREE = this.THREE;
            // Cone defaults: apex at +Y. We want apex along -Z (forward
            // direction in three.js camera convention) so that lookAt(target)
            // points the apex at the target.
            const geom = new THREE.ConeGeometry(0.006, 0.018, 8);
            geom.rotateX(-Math.PI / 2);

            // Add a tiny cube behind the cone for the "camera body".
            const body = new THREE.BoxGeometry(0.010, 0.010, 0.008);
            body.translate(0, 0, 0.012);

            // Merge both geometries so we can use a single InstancedMesh.
            // (BufferGeometryUtils isn't loaded as a separate script, so
            // we do it manually by appending attributes.)
            const merged = this._mergeGeoms(geom, body);

            const mat = new THREE.MeshBasicMaterial({ color });
            const mesh = new THREE.InstancedMesh(merged, mat, positions.length);
            const dummy = new THREE.Object3D();
            const tv = new THREE.Vector3(...target);
            for (let i = 0; i < positions.length; i++) {
                dummy.position.set(...positions[i]);
                dummy.lookAt(tv);
                dummy.updateMatrix();
                mesh.setMatrixAt(i, dummy.matrix);
            }
            mesh.instanceMatrix.needsUpdate = true;
            scene.add(mesh);
        }

        /**
         * LED-icon scatter: a small flat disc per light, oriented so its
         * +Z face points at the turntable centre (emitting surface).
         */
        _addLEDIcons(scene, positions, color, target) {
            const THREE = this.THREE;
            const geom = new THREE.CylinderGeometry(0.009, 0.009, 0.002, 20);
            // Cylinder axis defaults to +Y; rotate so axis is along +Z and
            // the disc face points along -Z (toward target after lookAt).
            geom.rotateX(Math.PI / 2);

            const mat = new THREE.MeshBasicMaterial({ color });
            const mesh = new THREE.InstancedMesh(geom, mat, positions.length);
            const dummy = new THREE.Object3D();
            const tv = new THREE.Vector3(...target);
            for (let i = 0; i < positions.length; i++) {
                dummy.position.set(...positions[i]);
                dummy.lookAt(tv);
                dummy.updateMatrix();
                mesh.setMatrixAt(i, dummy.matrix);
            }
            mesh.instanceMatrix.needsUpdate = true;
            scene.add(mesh);
        }

        /**
         * Concatenate two BufferGeometries on the position+normal+uv
         * attributes (only what InstancedMesh needs for MeshBasicMaterial).
         * Avoids pulling in three.js's BufferGeometryUtils as a separate
         * script.
         */
        _mergeGeoms(a, b) {
            const THREE = this.THREE;
            const out = new THREE.BufferGeometry();
            const concat = (name, components) => {
                const arrA = a.getAttribute(name).array;
                const arrB = b.getAttribute(name).array;
                const combined = new Float32Array(arrA.length + arrB.length);
                combined.set(arrA, 0);
                combined.set(arrB, arrA.length);
                out.setAttribute(name, new THREE.BufferAttribute(combined, components));
            };
            concat("position", 3);
            if (a.getAttribute("normal") && b.getAttribute("normal")) concat("normal", 3);
            if (a.getAttribute("uv") && b.getAttribute("uv")) concat("uv", 2);
            // Index: shift b's indices, concat
            const ia = a.getIndex(), ib = b.getIndex();
            if (ia && ib) {
                const aLen = ia.count, bLen = ib.count;
                const aVertCount = a.getAttribute("position").count;
                const idx = new Uint32Array(aLen + bLen);
                for (let i = 0; i < aLen; i++) idx[i] = ia.getX(i);
                for (let i = 0; i < bLen; i++) idx[aLen + i] = ib.getX(i) + aVertCount;
                out.setIndex(new THREE.BufferAttribute(idx, 1));
            }
            return out;
        }

        /**
         * Build the material bounding-box from bbox.json (world0 frame).
         * We use a Group so position + quaternion can be set in one go from
         * setMaterialTurnAngle.
         */
        _addMaterialBox(scene, bbox) {
            const THREE = this.THREE;
            const mn = bbox.bbox_min;
            const mx = bbox.bbox_max;
            const size = [mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]];
            const cen  = [(mn[0]+mx[0])/2, (mn[1]+mx[1])/2, (mn[2]+mx[2])/2];

            // A single solid-coloured slab. The wireframe edges we used
            // to draw (EdgesGeometry of the box) made the slab look like
            // it had non-uniform dark stripes from any angle that wasn't
            // perfectly axis-aligned, and dramatically worse from the
            // inside — so they're gone.
            const solidGeom = new THREE.BoxGeometry(size[0], size[1], size[2]);
            const solid = new THREE.Mesh(
                solidGeom,
                new THREE.MeshBasicMaterial({
                    color: 0xcfb86e,
                    side: THREE.FrontSide, // hide backfaces if camera goes inside
                }),
            );

            const group = new THREE.Group();
            group.add(solid);
            group.position.set(...cen);
            scene.add(group);

            // Remember the world0 centre — setMaterialTurnAngle uses this
            // to compute the rotated pose.
            this.materialBox       = group;
            this.materialBoxCenter = cen;
            // Selection rays should end at the TOP surface of the material
            // (where light reflects and what cameras look at). In base1
            // we rotate this point together with the box.
            this._materialTopCenter0 = [cen[0], cen[1], mx[2]];
            this._rayEndpoint        = [...this._materialTopCenter0];
        }

        /**
         * Rotate the material box by +turnAngleDeg around the calibrated
         * turntable axis, anchored at the turntable centre. Stored world0
         * pose is used as the zero-rotation reference.
         *
         * No-op when this scene is the world0 panel (rotation flag is off).
         */
        /**
         * Recolour the material slab. Pass an [r,g,b] in 0..255.
         * Used by obj.js once a frame has been loaded and the mean colour
         * of the HDR image is known.
         */
        /**
         * Swap the sphere selection placeholders for full 3D models (one
         * per marker). Each model is expected to be a THREE.Object3D
         * (typically a Group wrapping the loaded glTF scene). Position +
         * orientation are then driven by setSelection() exactly like the
         * spheres were.
         */
        attachSelectionModels(camModel, lightModel) {
            const swap = (oldMarker, newModel) => {
                if (!newModel) return oldMarker;
                if (oldMarker) {
                    // Copy the current world position so the new model
                    // appears where the placeholder was, then drop the
                    // old one from the scene.
                    newModel.position.copy(oldMarker.position);
                    this.scene.remove(oldMarker);
                }
                this.scene.add(newModel);
                return newModel;
            };
            this.selCam   = swap(this.selCam,   camModel);
            this.selLight = swap(this.selLight, lightModel);
        }

        setMaterialColor(rgb255) {
            if (!this.materialBox) return;
            // Only child is the solid slab now (wireframe removed).
            const solid = this.materialBox.children[0];
            if (!solid || !solid.material) return;
            solid.material.color.setRGB(
                rgb255[0] / 255.0,
                rgb255[1] / 255.0,
                rgb255[2] / 255.0,
            );
        }

        setMaterialTurnAngle(turnAngleDeg) {
            if (!this.materialBox || !this._rotateMaterialBox) return;
            const THREE = this.THREE;
            const C = window.CONFIG.CALIB;

            const axis = new THREE.Vector3(...C.TURNTABLE_AXIS).normalize();
            const q = new THREE.Quaternion().setFromAxisAngle(
                axis,
                turnAngleDeg * Math.PI / 180.0,
            );
            const ttc = new THREE.Vector3(...C.TURNTABLE_CENTER);
            const cen0 = new THREE.Vector3(...this.materialBoxCenter);

            // Rotate (cen0 - turntable_centre) by q, then put back relative
            // to the turntable centre. This matches T_{w0→b1}(turn_angle).
            const newPos = cen0.clone().sub(ttc).applyQuaternion(q).add(ttc);
            this.materialBox.position.copy(newPos);
            this.materialBox.quaternion.copy(q);

            // Also rotate the ray endpoint (material top centre) so that
            // dashed lines from the selected camera/light land on the
            // physical surface, not the calibrated rotation axis.
            const tc0 = new THREE.Vector3(...this._materialTopCenter0);
            const tcRot = tc0.clone().sub(ttc).applyQuaternion(q).add(ttc);
            this._rayEndpoint = [tcRot.x, tcRot.y, tcRot.z];
        }

        _addMarker(scene, color, radius, initialPos) {
            const THREE = this.THREE;
            const m = new THREE.Mesh(
                new THREE.SphereGeometry(radius, 16, 16),
                new THREE.MeshBasicMaterial({ color }),
            );
            m.position.set(...initialPos);
            scene.add(m);
            return m;
        }

        _addRay(scene, color) {
            const THREE = this.THREE;
            const g = new THREE.BufferGeometry();
            g.setAttribute(
                "position",
                new THREE.Float32BufferAttribute([0,0,0, 0,0,0], 3),
            );
            const ray = new THREE.Line(
                g,
                new THREE.LineDashedMaterial({
                    color, dashSize: 0.014, gapSize: 0.006,
                    transparent: true, opacity: 0.55,
                }),
            );
            ray.computeLineDistances();
            scene.add(ray);
            return ray;
        }

        /**
         * Update selected camera + light positions.
         * Pass null to leave a marker unchanged.
         *
         * Dashed rays end at the material's top-surface centre (rotated
         * along with the slab in base1; static in world0). Falls back to
         * the turntable centre if no bbox was supplied.
         */
        setSelection({ camPos, lightPos }) {
            const THREE = this.THREE;
            const endpoint = this._rayEndpoint || this._center;
            const endpointVec = new THREE.Vector3(...endpoint);
            if (camPos) {
                this.selCam.position.set(...camPos);
                // Aim the camera's lens (local +Z) at the material top
                // centre. For a sphere placeholder this is a no-op.
                this.selCam.lookAt(endpointVec);
                this._updateRay(this.camRay, camPos, endpoint);
            }
            if (lightPos) {
                this.selLight.position.set(...lightPos);
                // Aim the LED panel's emitting face (local +Z) at the
                // material top centre, just like the camera.
                this.selLight.lookAt(endpointVec);
                this._updateRay(this.lightRay, lightPos, endpoint);
            }
        }

        _updateRay(ray, from, to) {
            const THREE = this.THREE;
            const g = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(...from),
                new THREE.Vector3(...to),
            ]);
            ray.geometry.dispose();
            ray.geometry = g;
            ray.computeLineDistances();
        }

        resize() {
            const w = this.canvas.clientWidth;
            const h = this.canvas.clientHeight;
            if (w === 0 || h === 0) return;
            this.renderer.setSize(w, h, false);
            this.camera.aspect = w / h;
            this.camera.updateProjectionMatrix();
        }
    }

    window.Viz = { CombinedCaptureScene };
})();
